"""Whisper Subtitle API - membuat subtitle (SRT/VTT) dari rekaman percakapan atau lagu (MP3)."""

import logging
import os
import re
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path


def _register_cuda_dlls() -> None:
    """Windows: DLL cuBLAS/cuDNN dari paket pip nvidia-* harus didaftarkan sebelum ctranslate2 dimuat."""
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if os.name != "nt" or spec is None:
        return
    for root in spec.submodule_search_locations or []:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


_register_cuda_dlls()

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
import numpy as np
from faster_whisper import WhisperModel, decode_audio

MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3-turbo")
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
# int8_float16 di GPU: VRAM sekitar separuh float16, cukup untuk GPU 4 GB
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16" if DEVICE == "cuda" else "int8")
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"}
MODELS_DIR = Path(__file__).parent / "models"

# Model yang bisa dipilih per request: semua folder models/<nama> yang berisi model.bin,
# ditambah model default (diunduh otomatis jika belum ada di folder models).
AVAILABLE_MODELS = sorted(
    {p.parent.name for p in MODELS_DIR.glob("*/model.bin")} | {MODEL_SIZE}
)
ModelName = Enum("ModelName", {name: name for name in AVAILABLE_MODELS}, type=str)

# Baris subtitle dibuang jika level vokal (hasil demucs) di bawah ini, relatif
# terhadap bagian vokal terkeras lagu. Lirik asli umumnya -1..-10 dB,
# halusinasi di bagian instrumental sekitar -15..-30 dB.
VOCAL_MIN_DB = float(os.getenv("VOCAL_MIN_DB", "-13"))

# Batas satu baris subtitle
MAX_CHARS = 80
MAX_DURATION = 7.0

# Kalimat "hantu" yang sering dikarang Whisper pada bagian musik/hening
# (berasal dari data latih subtitle YouTube).
HALLUCINATION_RE = re.compile(
    r"(terima kasih|makasih)\s+(telah|sudah|udah)\s+menonton"
    r"|thanks?\s+(you\s+)?for\s+watching"
    r"|subscribe"
    r"|amara\.org"
    r"|subtitles?\s+(by|oleh)"
    r"|(diterjemahkan|terjemahan|teks|subtitle)\s+oleh",
    re.IGNORECASE,
)

app = FastAPI(title="Whisper Subtitle API", version="1.0.0")

# Log debug tiap request (DEBUG_REQUESTS=0 untuk mematikan): membantu melihat
# apa yang sebenarnya dikirim aplikasi lain ke API.
DEBUG_REQUESTS = os.getenv("DEBUG_REQUESTS", "1") != "0"
# DEBUG_SAVE_UPLOADS=1: simpan audio yang diterima (setelah dekripsi) ke folder debug_uploads
DEBUG_SAVE_DIR = Path(__file__).parent / "debug_uploads" if os.getenv("DEBUG_SAVE_UPLOADS") == "1" else None
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whisper-api")


def _describe_field(value) -> str:
    if isinstance(value, StarletteUploadFile):
        return f"<file {value.filename!r} content_type={value.content_type!r}>"
    return repr(value)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if not DEBUG_REQUESTS:
        return await call_next(request)
    start = time.perf_counter()
    log.info(
        "-> %s %s dari %s | content-type=%r content-length=%s",
        request.method,
        request.url.path,
        request.client.host if request.client else "?",
        request.headers.get("content-type"),
        request.headers.get("content-length"),
    )
    response = await call_next(request)
    log.info("<- %s %s %d (%.1f s)", request.method, request.url.path, response.status_code, time.perf_counter() - start)
    return response


@app.exception_handler(HTTPException)
async def log_http_error(request: Request, exc: HTTPException):
    log.warning("%d %s | %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """422: field wajib tidak ada atau nilainya tidak valid. Catat field yang diterima."""
    body = exc.body
    received = (
        {k: _describe_field(v) for k, v in body.multi_items()} if hasattr(body, "multi_items") else body
    )
    log.warning("422 %s | field diterima: %s | error: %s", request.url.path, received, exc.errors())
    errors = [
        {k: e[k] for k in ("type", "loc", "msg", "input") if k in e and not isinstance(e.get(k), StarletteUploadFile)}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})

_models: dict[str, WhisperModel] = {}
_model_lock = threading.Lock()


def resolve_model_path(name: str) -> str:
    """Pakai model di folder models/<nama> jika ada, jika tidak unduh dari Hugging Face."""
    local = MODELS_DIR / name
    return str(local) if (local / "model.bin").is_file() else name


def get_model(name: str = MODEL_SIZE) -> WhisperModel:
    with _model_lock:
        if name not in _models:
            _models[name] = WhisperModel(
                resolve_model_path(name),
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=os.cpu_count() or 4,
            )
        return _models[name]


_separator = None
_separator_lock = threading.Lock()


def get_separator():
    global _separator
    with _separator_lock:
        if _separator is None:
            import torch
            from demucs.pretrained import get_model as get_demucs_model

            torch.set_num_threads(os.cpu_count() or 4)
            _separator = get_demucs_model(DEMUCS_MODEL)
            _separator.eval()
    return _separator


def require_audio(samples: np.ndarray) -> None:
    """File yang tidak bisa di-decode (mis. masih terenkripsi) menghasilkan audio kosong."""
    if samples.size == 0:
        raise HTTPException(
            400,
            "Audio tidak bisa dibaca. Jika file terenkripsi dari recorder, isi parameter key; "
            "jika key sudah diisi, kemungkinan key salah",
        )


def extract_vocals(path: str) -> np.ndarray:
    """Pisahkan vokal dari musik dengan demucs. Hasil: audio mono 16 kHz untuk Whisper."""
    import julius
    import torch
    from demucs.apply import apply_model

    model = get_separator()
    sr = model.samplerate
    left, right = decode_audio(path, sampling_rate=sr, split_stereo=True)
    require_audio(left)
    wav = torch.from_numpy(np.stack([left, right]))

    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], device="cpu", split=True, overlap=0.25)[0]
    vocals = (sources[model.sources.index("vocals")] * std + mean).mean(0)
    return julius.resample_frac(vocals, sr, 16000).numpy().astype(np.float32)


def _timestamp(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def split_segment(words) -> list[dict]:
    """Pecah satu segmen menjadi baris subtitle pendek berdasarkan timestamp per kata."""
    groups, cur = [], []
    for w in words:
        if cur:
            text = "".join(x.word for x in cur + [w]).strip()
            if len(text) > MAX_CHARS or w.end - cur[0].start > MAX_DURATION:
                # Potong di koma terakhir agar frasa tidak terbelah
                cut = next(
                    (i + 1 for i in range(len(cur) - 1, 0, -1) if cur[i].word.strip().endswith(",")),
                    len(cur),
                )
                groups.append(cur[:cut])
                cur = cur[cut:]
        cur.append(w)
        if w.word.strip().endswith((".", "?", "!")):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return [
        {"start": g[0].start, "end": g[-1].end, "text": "".join(x.word for x in g).strip()}
        for g in groups
    ]


def build_segments(segments_iter) -> list[dict]:
    result = []
    for s in segments_iter:
        if HALLUCINATION_RE.search(s.text):
            continue
        if s.words:
            result.extend(split_segment(s.words))
        else:
            result.append({"start": s.start, "end": s.end, "text": s.text.strip()})
    return [seg for seg in result if seg["text"]]


def drop_silent_vocals(segments: list[dict], vocals: np.ndarray, sr: int = 16000) -> list[dict]:
    """Buang baris yang muncul saat track vokal hampir hening (halusinasi di bagian instrumental)."""
    frame = int(0.05 * sr)
    rms = np.sqrt(np.convolve(vocals**2, np.ones(frame) / frame, mode="same"))
    ref = np.percentile(rms, 95) + 1e-9
    kept = []
    for seg in segments:
        part = rms[int(seg["start"] * sr) : max(int(seg["end"] * sr), int(seg["start"] * sr) + 1)]
        if 20 * np.log10(np.median(part) / ref + 1e-9) >= VOCAL_MIN_DB:
            kept.append(seg)
    return kept


def to_srt(segments: list[dict]) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{_timestamp(seg['start'], ',')} --> {_timestamp(seg['end'], ',')}\n{seg['text']}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT\n"]
    for seg in segments:
        lines.append(
            f"{_timestamp(seg['start'], '.')} --> {_timestamp(seg['end'], '.')}\n{seg['text']}\n"
        )
    return "\n".join(lines)


# --- Mode percakapan ---------------------------------------------------------

class Mode(str, Enum):
    percakapan = "percakapan"
    lagu = "lagu"


class ChannelMode(str, Enum):
    otomatis = "otomatis"
    mono = "mono"
    terpisah = "terpisah"


# VAD longgar: bagian bising tanpa ucapan dilewati (mencegah halusinasi),
# jeda pendek antar kalimat tetap dianggap satu bagian.
VAD_SPEECH = dict(threshold=0.3, min_silence_duration_ms=1500, speech_pad_ms=500)


def decrypt_aes_cbc(data: bytes, key: str) -> bytes:
    """Dekripsi file recorder: AES-128-CBC, IV = key, padding PKCS#7."""
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    k = key.encode()
    if len(k) != 16:
        raise HTTPException(400, "Key AES-128 harus 16 karakter")
    if len(data) % 16:
        raise HTTPException(400, "File bukan data AES-CBC yang valid (ukuran bukan kelipatan 16)")
    decryptor = Cipher(algorithms.AES(k), modes.CBC(k)).decryptor()
    plain = decryptor.update(data) + decryptor.finalize()
    try:
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(plain) + unpadder.finalize()
    except ValueError:
        raise HTTPException(400, "Dekripsi gagal: key salah atau file tidak terenkripsi")


def squelch_transmissions(x: np.ndarray, sr: int = 16000) -> list[tuple[int, int]]:
    """Rekaman radio/HT dengan squelch: di antara transmisi audionya hening digital total.
    Setiap blok bersuara (min 0,5 s) di antara hening itu dianggap transmisi. Rekaman tanpa
    hening digital minimal 1 detik (mis. rekaman ruangan ber-noise) tidak menghasilkan apa-apa."""
    frame = sr // 100  # 10 ms
    n = len(x) // frame
    if n == 0:
        return []
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    silent = rms < 10 ** (-70 / 20)
    if silent.sum() < 100:
        return []
    blocks, start = [], None
    for i, s in enumerate(np.append(silent, True)):
        if not s and start is None:
            start = i
        elif s and start is not None:
            if i - start >= 50:
                blocks.append((start * frame, i * frame))
            start = None
    return blocks


def detect_speech(x: np.ndarray) -> list[tuple[int, int]]:
    """Rentang ucapan (dalam sampel) dari Silero VAD. VAD dijalankan pada dua level lalu digabung:
    Silero sensitif terhadap level, ucapan radio/HT yang sangat keras dan clipping hampir tidak
    terdeteksi pada level penuh tapi terdeteksi setelah diturunkan 20 dB (dan sebaliknya
    untuk rekaman biasa)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    x = prepare_speech(x)
    chunks = sorted(
        [
            (c["start"], c["end"])
            for level in (x, x * 0.1)
            for c in get_speech_timestamps(level, VadOptions(**VAD_SPEECH))
        ]
        + squelch_transmissions(x)
    )
    merged: list[list[int]] = []
    for start, end in chunks:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def has_speech_activity(x: np.ndarray, sr: int = 16000) -> bool:
    """Channel yang hanya berisi desis/noise latar tidak punya bagian ucapan minimal 1 detik."""
    return sum(e - s for s, e in detect_speech(x)) >= sr


def prepare_speech(x: np.ndarray) -> np.ndarray:
    """Normalisasi volume (penguatan maks 30 dB) agar rekaman pelan tetap terbaca.
    Catatan: high-pass filter sengaja tidak dipakai; pada uji rekaman recorder,
    kombinasinya dengan normalisasi membuat Whisper melewatkan kalimat."""
    peak = np.percentile(np.abs(x), 99.9) + 1e-9
    gain = min(0.9 / peak, 10 ** (30 / 20))
    return np.clip(x * gain, -1, 1).astype(np.float32)


def speech_clips(x: np.ndarray, sr: int = 16000) -> list[float]:
    """Rentang ucapan dari VAD untuk clip_timestamps Whisper. Setiap rentang didekode sendiri,
    tidak disambung seperti vad_filter bawaan (sambungan itu membuat Whisper melewatkan
    kalimat yang mirip/berulang). Rentang berdekatan (<2 detik) digabung, maks 30 detik."""
    merged: list[list[float]] = []
    for s, e in detect_speech(x):
        start, end = s / sr, e / sr
        if merged and start - merged[-1][1] < 2.0 and end - merged[-1][0] <= 30:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [t for pair in merged for t in pair]


def split_channels(
    path: str, how: ChannelMode = ChannelMode.otomatis
) -> list[tuple[str | None, np.ndarray]]:
    """otomatis: stereo dengan isi berbeda per channel (mis. satu channel per perangkat/pembicara)
    ditranskrip per channel dan channel yang hanya berisi desis dilewati; selain itu digabung mono.
    mono: selalu digabung. terpisah: kedua channel selalu ditranskrip sendiri-sendiri."""
    left, right = decode_audio(path, split_stereo=True)
    require_audio(left)
    is_mono_file = np.array_equal(left, right)
    if how == ChannelMode.terpisah and not is_mono_file:
        return [("CH1", left), ("CH2", right)]
    if (
        how == ChannelMode.otomatis
        and not is_mono_file
        and left.std() > 0
        and right.std() > 0
        and abs(np.corrcoef(left, right)[0, 1]) < 0.5
    ):
        channels = [(f"CH{i}", ch) for i, ch in ((1, left), (2, right)) if has_speech_activity(ch)]
        if channels:
            return channels
    return [(None, (left + right) / 2)]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": MODEL_SIZE,
        "available_models": AVAILABLE_MODELS,
        "loaded_models": sorted(_models),
        "device": DEVICE,
        "demucs_model": DEMUCS_MODEL,
        "demucs_loaded": _separator is not None,
    }


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(..., description="File audio (mp3, wav, m4a, ...)"),
    format: str = Form("srt", description="srt | vtt | json | txt"),
    model: ModelName = Form(
        ModelName(MODEL_SIZE),
        description=f"Jika tidak dipilih/kosong, dipakai {MODEL_SIZE} (default). "
        "large-v3-turbo: lebih cepat dan cakupan lagu lebih lengkap. "
        "large-v3: kata lebih akurat, sekitar 1,7x lebih lambat",
    ),
    mode: Mode = Form(
        Mode.percakapan,
        description="percakapan (default): rekaman bicara, termasuk yang ada noise latar; "
        "stereo dengan isi berbeda per channel ditranskrip per channel ([CH1]/[CH2]). "
        "lagu: vokal dipisahkan dari musik dengan demucs",
    ),
    key: str | None = Form(
        None, description="Isi jika file terenkripsi AES-128-CBC dari recorder (16 karakter)"
    ),
    language: str | None = Form(None, description="Kode bahasa, mis. 'id' atau 'en'. Kosong = deteksi otomatis"),
    task: str = Form("transcribe", description="transcribe | translate (ke bahasa Inggris)"),
    prompt: str | None = Form(
        None, description="Konteks/kosakata untuk membantu Whisper, mis. nama orang atau istilah"
    ),
    vad: bool | None = Form(
        None,
        description="Lanjutan. Kosong = ikut mode (percakapan: aktif, lagu: mati)",
    ),
    channels: ChannelMode = Form(
        ChannelMode.otomatis,
        description="Hanya untuk mode percakapan. otomatis (default): stereo yang isi tiap channel-nya "
        "berbeda ditranskrip per channel ([CH1]/[CH2]), selain itu digabung. "
        "mono: selalu digabung, sekitar 2x lebih cepat tapi ucapan bersamaan bisa tercampur/hilang. "
        "terpisah: kedua channel selalu ditranskrip sendiri-sendiri",
    ),
    separate_vocals: bool | None = Form(
        None,
        description="Lanjutan. Kosong = ikut mode (percakapan: mati, lagu: aktif)",
    ),
):
    fmt = format.lower()
    if fmt not in {"srt", "vtt", "json", "txt"}:
        raise HTTPException(400, "format harus srt, vtt, json, atau txt")
    if task not in {"transcribe", "translate"}:
        raise HTTPException(400, "task harus transcribe atau translate")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Ekstensi tidak didukung: {ext or '(kosong)'}")

    data = file.file.read()
    if DEBUG_REQUESTS:
        log.info(
            "transcribe: file=%r (%d byte, content_type=%r) format=%s model=%s mode=%s key=%s "
            "language=%r task=%s prompt=%r vad=%s channels=%s separate_vocals=%s",
            file.filename, len(data), file.content_type, fmt, model.value, mode.value,
            f"<{len(key)} karakter>" if key else None,
            language, task, prompt, vad, channels.value, separate_vocals,
        )
    if not data:
        raise HTTPException(400, "File kosong (0 byte)")
    if key:
        data = decrypt_aes_cbc(data, key)
    if DEBUG_SAVE_DIR:
        DEBUG_SAVE_DIR.mkdir(exist_ok=True)
        (DEBUG_SAVE_DIR / f"{time.strftime('%H%M%S')}_{Path(file.filename).name}").write_bytes(data)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    is_song = mode == Mode.lagu
    use_demucs = is_song if separate_vocals is None else separate_vocals
    use_vad = (not is_song) if vad is None else vad
    options = dict(
        language=language or None,
        task=task,
        beam_size=5,
        initial_prompt=prompt or None,
        # Cegah halusinasi menular ke segmen berikutnya
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    if not use_vad:
        # Lewati jeda/instrumental panjang yang memicu halusinasi
        options.update(hallucination_silence_threshold=2.0)

    def run(audio: np.ndarray):
        """Transkrip audio 16 kHz; dengan VAD hanya rentang ucapan yang didekode."""
        opts = dict(options)
        if use_vad:
            clips = speech_clips(audio)
            if DEBUG_REQUESTS:
                log.info(
                    "VAD: %d rentang ucapan, total %.1f s dari %.1f s audio",
                    len(clips) // 2, sum(clips[1::2]) - sum(clips[::2]), len(audio) / 16000,
                )
            if not clips:
                return [], None
            opts["clip_timestamps"] = clips
        segments_iter, run_info = whisper.transcribe(audio, **opts)
        return build_segments(segments_iter), run_info

    try:
        whisper = get_model(model.value)
        if use_demucs:
            vocals = extract_vocals(tmp_path)
            segments, info = run(vocals)
            segments = drop_silent_vocals(segments, vocals)
        else:
            parts = split_channels(tmp_path, channels)
            segments, info = [], None
            if DEBUG_REQUESTS:
                log.info(
                    "channel: %s",
                    ", ".join(
                        f"{label or 'mono'} level {20 * np.log10(np.sqrt(np.mean(a**2)) + 1e-9):.1f} dBFS"
                        for label, a in parts
                    ),
                )
            for label, audio in parts:
                channel_segments, channel_info = run(prepare_speech(audio))
                info = info or channel_info
                for seg in channel_segments:
                    if len(parts) > 1:
                        seg["channel"] = label
                        seg["text"] = f"[{label}] {seg['text']}"
                    segments.append(seg)
            segments.sort(key=lambda s: s["start"])
    except HTTPException:
        raise
    except Exception as e:
        log.exception("transcribe gagal untuk %r", file.filename)
        hint = "" if key else " (jika file terenkripsi dari recorder, isi parameter key)"
        raise HTTPException(500, f"Gagal memproses audio{hint}: {e}")
    finally:
        os.unlink(tmp_path)

    if DEBUG_REQUESTS:
        log.info("hasil: %d baris subtitle untuk %r", len(segments), file.filename)
    base = Path(file.filename).stem
    if fmt == "json":
        return JSONResponse(
            {
                "model": model.value,
                "mode": mode.value,
                "channels": channels.value,
                # info kosong jika tidak ada ucapan sama sekali yang terdeteksi
                "language": info.language if info else None,
                "language_probability": info.language_probability if info else None,
                "duration": info.duration if info else None,
                "segments": segments,
            }
        )

    if fmt == "srt":
        body, media = to_srt(segments), "application/x-subrip"
    elif fmt == "vtt":
        body, media = to_vtt(segments), "text/vtt"
    else:
        body, media = "\n".join(s["text"] for s in segments), "text/plain"

    return PlainTextResponse(
        body,
        media_type=f"{media}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{base}.{fmt}"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8085")))
