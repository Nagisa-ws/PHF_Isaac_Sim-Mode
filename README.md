# Panduan singkat: memutar USD dan membuat video 4 kamera di Isaac Sim

Butuh: komputer dengan NVIDIA Isaac Sim dan GPU RTX.

## Persiapan

1. Ekstrak `mission3_potek_usd.zip`. Hasilnya folder `usd/`:
   ```
   usd/
     mission3.usdc         <- file yang dibuka
     mission3_meta.json    <- dipakai script video
     textures/             <- harus tetap di sebelah mission3.usdc
   ```
2. Untuk video: ambil `scripts/isaac_render_video.py` dari `isaac_replay_code.zip`.

## A. Memutar USD (kamera bebas)

1. Buka Isaac Sim.
2. **File > Open**, pilih `usd/mission3.usdc`, tunggu sampai selesai dimuat.
3. Tekan **Play** (atau Spasi). Geser timeline di bawah untuk melompat ke waktu lain.
4. Ganti kamera lewat menu kamera di pojok viewport:

| Kamera | Tampilan |
|---|---|
| Perspective | bebas: Alt + drag = putar, scroll = zoom |
| `cam_follow` | mengikuti kendaraan dari samping-belakang |
| `cam_corner` | dari sudut bak kendaraan ke area kerja |
| `cam_wrist` | dari pergelangan gripper |
| `overview` | dari depan-kiri lengan |

Durasi animasi: 21 menit 49 detik (25 fps).

## B. Video 4 kamera dalam satu frame

Jalankan dari folder instalasi Isaac Sim (folder yang berisi `python.bat` / `python.sh`).

**Windows**
```
python.bat C:\path\ke\isaac_render_video.py --usd C:\path\ke\usd\mission3.usdc --start 0 --duration 20
```

**Linux**
```
./python.sh /path/ke/isaac_render_video.py --usd /path/ke/usd/mission3.usdc --start 0 --duration 20
```

Hasil: `usd/mission3_4cam.mp4` (1920 x 1080):

```
Follow Vehicle | Corner Camera
Wrist Camera   | Overview
```
Baris bawah: waktu, fase, jumlah buah di keranjang.

Coba dulu potongan 20 detik seperti di atas. Kalau sudah benar, pilih salah satu:

| Tujuan | Tambahan pada perintah |
|---|---|
| Seluruh baris (bisa berjam-jam) | hapus `--start` dan `--duration` |
| Time lapse 4x | `--speed 4` |
| Potongan tertentu, mis. 30 detik mulai menit ke-1 | `--start 60 --duration 30` |
| Render lebih ringan | `--tile 640 360` |
| Hanya kamera tertentu | `--cameras follow wrist` |
| Nama file keluaran | `--out C:\path\video.mp4` |

## Kalau ada masalah

- **Benda tampak abu-abu / tanpa tekstur:** folder `textures/` tidak berada di sebelah `mission3.usdc`.
- **Scene terlalu gelap atau terang:** pilih `/World/lights/sky` atau `/World/lights/sun` di panel Stage, ubah `intensity`.
- **Script video error:** script ini belum pernah diuji di Isaac Sim (ditulis mengikuti dokumentasi Isaac Sim 6.x).
  Salin pesan error-nya dan kirimkan.
