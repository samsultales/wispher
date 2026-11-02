# Whisper Subtitle API

API Python (FastAPI + faster-whisper) untuk membuat subtitle dari rekaman percakapan atau lagu (MP3).
Tidak perlu ffmpeg: decoding audio sudah ditangani PyAV.

## Instalasi

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Menjalankan

```bash
.venv\Scripts\python app.py
```

Server berjalan di `http://localhost:8085`, dengan dokumentasi interaktif di `http://localhost:8085/docs`.
Model dibaca dari folder `models\<nama-model>` jika ada. Jika tidak ada, model diunduh otomatis saat request pertama (`large-v3-turbo` sekitar 1,6 GB).

Jika unduhan otomatis macet, unduh manual file `config.json`, `preprocessor_config.json`, `tokenizer.json`,
`vocabulary.json`, dan `model.bin` dari https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/tree/main
ke folder `models\large-v3-turbo\`.

### Konfigurasi (environment variable)

| Variabel | Default | Keterangan |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | Model default jika parameter `model` tidak diisi: `tiny`, `base`, `small`, `medium`, `large-v3-turbo`, `large-v3` |
| `WHISPER_DEVICE` | `cuda` | `cuda` (GPU NVIDIA) atau `cpu` |
| `WHISPER_COMPUTE_TYPE` | `int8_float16` (cuda), `int8` (cpu) | `float16` di GPU lebih presisi tapi butuh VRAM sekitar 2x |
| `DEMUCS_MODEL` | `htdemucs` | Model pemisah vokal untuk mode lagu: `htdemucs` atau `htdemucs_ft` (lebih bersih, sekitar 4x lebih lambat) |
| `VOCAL_MIN_DB` | `-13` | Mode lagu: baris dibuang jika level vokal hasil demucs di bawah nilai ini (filter halusinasi). Turunkan (mis. `-18`) jika ada lirik pelan yang ikut hilang |
| `PORT` | `8085` | Port server |
| `DEBUG_REQUESTS` | `1` | Log setiap request: field yang diterima, ukuran file, status, durasi, dan detail error. `0` untuk mematikan |
| `DEBUG_SAVE_UPLOADS` | `0` | `1`: simpan audio yang diterima (setelah dekripsi) ke folder `debug_uploads` untuk analisis |

## Endpoint

### `POST /transcribe` (multipart/form-data)

| Field | Wajib | Keterangan |
|---|---|---|
| `file` | ya | File audio (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.mp4`) |
| `format` | tidak | `srt` (default), `vtt`, `json`, `txt` |
| `model` | tidak | Dropdown berisi semua folder di `models\` yang punya `model.bin`. Kosong = `large-v3-turbo` (lebih cepat). `large-v3`: kata lebih akurat, sekitar 1,7x lebih lambat |
| `mode` | tidak | `percakapan` (default) atau `lagu`, lihat di bawah |
| `key` | tidak | Key 16 karakter jika file terenkripsi AES-128-CBC dari recorder |
| `channels` | tidak | Mode percakapan saja. `otomatis` (default), `mono`, atau `terpisah`, lihat di bawah |
| `language` | tidak | mis. `id`, `en`. Jika dikosongkan, bahasa dideteksi otomatis |
| `task` | tidak | `transcribe` (default) atau `translate` (terjemahkan ke Inggris) |
| `prompt` | tidak | Konteks atau kosakata untuk Whisper, mis. nama orang, nama produk, atau istilah |
| `vad` | tidak | Lanjutan. Kosong = ikut mode |
| `separate_vocals` | tidak | Lanjutan. Kosong = ikut mode |

#### Mode `percakapan` (default)

Untuk rekaman bicara, termasuk yang ada noise di latar belakang:

- VAD aktif: bagian yang hanya berisi noise dilewati, jadi Whisper tidak mengarang kalimat seperti "Terima kasih."
- Volume dinormalisasi (penguatan maksimal 30 dB) agar rekaman pelan tetap terbaca.
- Stereo yang isi tiap channel-nya berbeda (mis. satu channel per perangkat) ditranskrip per channel,
  dan setiap baris diberi label `[CH1]` atau `[CH2]`. Channel yang hanya berisi desis dilewati.
  Stereo biasa (kedua channel mirip) dan file mono digabung menjadi satu.
- Perilaku channel bisa dipaksa dengan parameter `channels`:

  | Nilai | Perilaku | Kapan dipakai |
  |---|---|---|
  | `otomatis` | Seperti dijelaskan di atas | Default, cocok untuk sebagian besar rekaman |
  | `mono` | Kedua channel selalu digabung | Butuh cepat (sekitar 2x) dan pembicara jarang bicara bersamaan. Ucapan bersamaan bisa tercampur, dan suara yang pelan bisa hilang |
  | `terpisah` | Kedua channel selalu ditranskrip sendiri-sendiri dengan label `[CH1]`/`[CH2]` | Deteksi otomatis keliru menggabungkan channel, mis. karena ada crosstalk antar mic |

  File mono asli selalu diproses sebagai mono.

#### Mode `lagu`

- Vokal dipisahkan dari musik dengan demucs (sekitar 2 menit untuk lagu 5 menit di CPU).
- VAD mati karena sering membuang nyanyian.
- Baris yang muncul saat track vokal hening dibuang (filter halusinasi di bagian instrumental).

Pada kedua mode, kalimat halusinasi khas Whisper seperti "Terima kasih telah menonton" dibuang, dan segmen panjang
dipecah menjadi baris subtitle maksimal sekitar 80 karakter atau 7 detik.

Contoh:

```bash
curl -X POST http://localhost:8085/transcribe -F "file=@rekaman.mp3" -F "language=id" -F "key=<key-recorder>" -o rekaman.srt
```

```bash
curl -X POST http://localhost:8085/transcribe -F "file=@lagu.mp3" -F "mode=lagu" -F "language=id" -o lagu.srt
```

### `GET /health`

Menampilkan status server dan model.
