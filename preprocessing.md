# Alur Kerja Skrip `preprocessing.py`

Dokumen ini menjelaskan alur kerja skrip `preprocessing.py`, yang bertanggung jawab untuk memproses file Markdown, membuat embedding, dan menyimpannya ke dalam database vektor Qdrant.

## Diagram Alur

```mermaid
graph TD
    A[Mulai] --> B{Inisialisasi Klien & Koneksi};
    B --> C{Cari File Markdown di './data'};
    C --> D{Iterasi Setiap File};
    D --> E[Baca & Ubah Markdown ke Teks Polos];
    E --> F[Bagi Teks Menjadi Bagian Kecil (Chunks)];
    F --> G[Kumpulkan Semua Chunks dengan ID & Sumber];
    D -- Loop untuk setiap file --> F;
    G --> H{Buat Embeddings untuk Semua Chunks};
    H --> I[Siapkan Data (PointStruct)];
    I --> J[Upsert ke Qdrant];
    J --> K[Selesai];

    subgraph "Inisialisasi"
        B
    end

    subgraph "Pemrosesan File"
        C
        D
        E
        F
    end

    subgraph "Embedding & Penyimpanan"
        G
        H
        I
        J
    end
```

## Penjelasan Langkah-demi-Langkah

1.  **Inisialisasi dan Konfigurasi**
    *   Skrip memuat variabel lingkungan dari file `.env`, termasuk kunci API untuk Google dan kredensial untuk Qdrant.
    *   Klien untuk Google Generative AI dan Qdrant diinisialisasi.
    *   Skrip memeriksa apakah koleksi Qdrant dengan nama yang ditentukan sudah ada. Jika tidak, koleksi baru akan dibuat dengan konfigurasi vektor yang sesuai (ukuran 768 dan jarak kosinus).

2.  **Pencarian File Markdown**
    *   Fungsi `main` menggunakan `glob` untuk secara rekursif menemukan semua file dengan ekstensi `.md` di dalam direktori `./data` dan subdirektorinya.

3.  **Pemrosesan Setiap File**
    *   Skrip melakukan iterasi pada setiap file Markdown yang ditemukan.
    *   **Membaca dan Membersihkan**: Konten file Markdown dibaca, dikonversi ke HTML, lalu diubah menjadi teks polos. Ini membantu menghilangkan sintaks Markdown.
    *   **Chunking**: Teks polos yang sudah bersih kemudian dipecah menjadi beberapa bagian yang lebih kecil (chunks) menggunakan fungsi `chunk_text`. Setiap chunk dibatasi hingga sekitar 300 token untuk memastikan ukurannya optimal untuk proses embedding.

4.  **Pengumpulan Chunks**
    *   Setiap chunk disimpan dalam sebuah list bersama dengan:
        *   **ID Unik**: Dibuat menggunakan `uuid.uuid4()`.
        *   **Teks**: Konten chunk itu sendiri.
        *   **Sumber**: Path relatif dari file asalnya.

5.  **Pembuatan Embedding**
    *   Setelah semua file diproses dan semua chunk dikumpulkan, fungsi `embed_texts` dipanggil.
    *   Fungsi ini mengirimkan teks dari semua chunk ke model embedding Google (`models/embedding-001`) untuk menghasilkan vektor representasi.
    *   Proses ini dilakukan secara batch untuk mengelola beban kerja dan menangani potensi kegagalan API dengan lebih baik.

6.  **Persiapan dan Penyimpanan Data**
    *   Data yang telah di-embed (vektor) dan metadata-nya (teks dan sumber) diformat ke dalam struktur `PointStruct` yang dibutuhkan oleh Qdrant.
    *   Terakhir, skrip menggunakan metode `upsert` dari klien Qdrant untuk memasukkan (atau memperbarui) semua titik data ini ke dalam koleksi yang telah ditentukan.

7.  **Selesai**
    *   Setelah data berhasil disimpan, skrip mencetak pesan konfirmasi.
