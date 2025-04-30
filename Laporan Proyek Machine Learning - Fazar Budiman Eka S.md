# Laporan Proyek Machine Learning - Fazar Budiman Eka S

## Project Overview
Indonesia merupakan negara kepulauan dengan lebih dari 17.000 pulau yang tersebar di seluruh penjuru nusantara. Kondisi geografis ini menjadikan Indonesia kaya akan destinasi wisata yang menarik bagi wisatawan domestik maupun mancanegara. Berdasarkan data Kementerian Pariwisata dan Ekonomi Kreatif (Kemenparekraf), tercatat sebanyak **927.746** wisatawan mancanegara berkunjung ke Indonesia pada Januari 2024, meningkat sebesar **16,19%** dibandingkan periode yang sama pada tahun sebelumnya [1]. Peningkatan ini menunjukkan potensi besar sektor pariwisata nasional. Namun, penggunaan teknologi dalam pengembangan sektor pariwisata di Indonesia masih tergolong minim. Oleh karena itu, diperlukan pengembangan sistem rekomendasi destinasi wisata yang bersifat personal untuk meningkatkan ketertarikan dan pengalaman wisatawan, sehingga dapat mendorong pertumbuhan kunjungan wisata ke Indonesia.

Untuk menjawab kebutuhan tersebut, akan dikembangkan sebuah sistem rekomendasi destinasi wisata berbasis *Collaborative Filtering*. Sistem ini bertujuan untuk memberikan rekomendasi destinasi berdasarkan pola preferensi pengguna lain yang memiliki kesamaan perilaku atau minat. Dalam implementasinya, sistem akan menggunakan Surprise[2], yaitu pustaka Python yang dirancang khusus untuk membangun dan menganalisis sistem rekomendasi berbasis *collaborative filtering* secara efisien dan fleksibel.

Proyek ini diharapkan dapat meningkatkan pengalaman pengguna, mendorong pertumbuhan sektor pariwisata, serta memanfaatkan pendekatan berbasis data untuk pengembangan wisata di Indonesia.


## Business Understanding
### Problem Statements
Berdasarkan latar belakang diatas, maka dirumuskan masalah sebagai berikut:
- Bagaimana membangun sistem rekomendasi yang mampu memberikan rekomendasi destinasi wisata kepada para wisatawan dengan berbasis *collaborative filtering*?
- Dengan menggunakan pustaka Surprise, algoritma apa yang paling optimal dalam sistem rekomendasi berbasis *collaborative filtering*?

### Goals
Adapun tujuan dari proyek ini adalah sebagai berikut:
- Membangun sistem rekomendasi berbasis *collaborative filtering* untuk memberikan rekomendasi destinasi kepada para wisatawan.
- Menentukan algoritma yang paling optimal dalam sistem rekomendasi berbasis *collaborative filtering*.

### Solution statements
- Mengimplementasikan algoritma Slope One dalam pustaka Surprise sebagai untuk mencapai tujuan dari proyek ini.
- Mengimplementasikan algoritma Non-negative Matrix Factorization (NMF) dan Co Clustering sebagai algoritma pembanding sekaligus menerapkan Grid Search untuk menentukan paramater terbaik dalam melatih modelnya.


## Data Understanding
Dataset yang digunakan berjudul **Indonesia Tourism Destination**. Dataset ini berisi daftar wisata yang tersebar di 5 kota besar Indonesia, kota tersebut meliputi Jakarta, Yogyakarta, Semarang, Bandung dan Surabaya. Adapun beberapa detail informasinya sebagai berikut:
- Tautan dataset : [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)
- Dataset ini terdiri dari 4 file, yaitu tourism_with_id.csv, user.csv, tourism_rating.csv dan package_tourism.
- Adapun variabel dalam file tourism_with_id.csv meliputi:
    -  Place_Id: Merupakan id dari setiap destinasi.
    -  Place_Name: Merupakan nama dari setiap destinasi.
    -  Description: Deskripsi singkat tentang destinasi wisata, berisi informasi penting atau daya tarik tempat tersebut.
    -  Category: Jenis kategori dari destinasi tersebut.
    -  City: Kota atau lokasi administratif di mana destinasi wisata tersebut berada.
    -  Price: Harga tiket masuk ke destinasi wisata (dalam satuan Rupiah).
    -  Rating: Skor ulasan atau penilaian dari pengguna terhadap destinasi tersebut (biasanya dalam skala 1–5).
    -  Time_Minutes: Estimasi waktu (dalam menit) yang dibutuhkan untuk mengeksplorasi destinasi wisata tersebut.
    -  Coordinate: Koordinat geografis destinasi dalam bentuk dictionary {lat: ..., lng: ...} (latitude dan longitude).
    -  Lat: Nilai latitude (garis lintang) dari lokasi destinasi wisata, biasanya dalam derajat desimal.
- Adapun variabel dalam file user.csv meliputi:
    - User_Id: ID unik dari setiap wisatawan.
    - Location: Merupakan kota tempat tinggal wisatawan.
    - Age: Umur dari wisatawan.
- Adapun variabel dalam file tourism_rating.csv meliputi:   
    - User_Id: ID unik dari setiap wisatawan.
    - Place_Id: ID unik dari destinasi wisata.
    - Place_Ratings: Rating yang diberikan wisatawan kepada destinasi wisata dengan skala 1-5.
- Adapun variabel dalam file package_tourism meliputi:
    - Package: ID atau nomor unik dari paket wisata yang ditawarkan, biasanya berisi kombinasi beberapa destinasi wisata.
    - City: Nama kota tempat paket wisata tersebut berada atau berfokus.
    - Place_Tourism1: Destinasi wisata pertama yang termasuk dalam paket wisata.
    - Place_Tourism2: Destinasi wisata kedua yang termasuk dalam paket wisata. (Bisa berisi null jika tidak ada).
    - Place_Tourism3: Destinasi wisata ketiga dalam paket wisata. (Bisa berisi null jika tidak ada).
    - Place_Tourism4: Destinasi wisata keempat dalam paket wisata. (Bisa berisi null jika tidak ada).
    - Place_Tourism5: Destinasi wisata kelima dalam paket wisata. (Bisa berisi null jika tidak ada).

### Exploratory Data Analysis
Pada proyek ini hanya menggunakan 2 file dari dataset tersebut yaitu tourism_with_id.csv dan tourism_rating.csv. Adapun informasi detail dalam dataset adalah sebagai berikut:
- place_rating.csv
    Dengan menggunakan fungsi `ratings.describe()` didapat informasi bahwa jumlah rating yang diberikan dari 300 wisatawan kepada 437 destinasi berjumlah 10.000. Adapun rating yang diberikan memiliki skala 1-5. Artinya seluruh wisatawan telah memiliki data historis begitu pun destinasi telah mendapatkan interaksi dari wisatawan.
- tourism_with_id.csv
    ![Sebaran Destinasi](https://i.imgur.com/5ygkpI8.png)
    Berdasarkan grafik diatas terlihat bahwa sebaran di masing-masing kota tidak seimbang lebih banyak destinasinya di kota Yogyakarta dan Bandung dibanding kota lainnya. Begitupun jenis kategorinya banyak di kategori Taman Hiburan, Budaya dan Cagar Alam tapi sedikit di kategori Bahari, Tempat Ibadah dan Pusat Perbelanjaan.
    

## Data Preparation
Agar model dapat dengan optimal mempelajari dataset, ada beberapa hal yang dilakukan dalam tahap ini diantaranya:
- **Mengubah format dataset menjadi format dataset internal Surprise**. Hal ini dilakukan karena dalam implementasi modelnya menggunakan pustaka Surprise sehingga datasetnya pun perlu diubah agar model dapat dilatih dengan dataset tersebut.
- **Data Splitting** yaitu untuk membagi data menjadi data latih dan data uji. Adapun proporsinya yaitu 70% untuk data latih dan 30% untuk data uji. Hal ini dilakukan untuk memisahkan data yang menjadi data latih model dan data uji model.


## Modeling
Dalam proyek ini dataset diuji coba dengan **3 algoritma** sistem rekomendasi berbasis *collaborative filtering* untuk menentukan algoritma yang paling optimal memberikan rekomendasi item.
- **Slope One**
    Slope One adalah algoritma *collaborative filtering* berbasis item yang bekerja dengan menghitung rata-rata selisih rating antar pasangan item. Ketika ingin memprediksi rating untuk item yang belum diberi rating oleh pengguna, algoritma ini menggunakan rating pengguna terhadap item lain dan menyesuaikannya berdasarkan rata-rata selisih yang telah dihitung. Proses ini sederhana, cepat, dan efisien karena hanya membutuhkan perhitungan selisih rata-rata tanpa training model yang kompleks, sehingga cocok untuk sistem rekomendasi yang dinamis dan skalabel. Dalam implementasinya algoritma ini tidak menggunakan parameter apapun sehingga cukup seperti ini:
    `model_slope_one = SlopeOne()`

- **Non-negative Matrix Factorization**
    NMF (Non-negative Matrix Factorization) adalah algoritma berbasis model yang digunakan dalam *collaborative filtering* untuk sistem rekomendasi. NMF bekerja dengan mendekonstruksi matriks user-item (yang berisi rating) menjadi dua matriks lebih kecil — satu matriks untuk user dan satu untuk item — dengan syarat seluruh elemen bernilai non-negatif. Proses ini memungkinkan sistem untuk menemukan pola laten (hubungan tersembunyi) antara pengguna dan item, sehingga dapat memprediksi rating yang belum ada. Karena hanya menghasilkan nilai positif, NMF sering menghasilkan hasil yang lebih stabil dan interpretatif dibandingkan metode dekomposisi lainnya. Dengan menggunakan GridSearchCV untuk menentukan parameter terbaik dalam algoritma ini, sehingga implementasinya menjadi seperti ini: 
    `model_NMF = NMF(n_factors=100, n_epochs=50, reg_pu=0.05, reg_qi=0.05)`

    Pada kode tersebut, model NMF (Non-negative Matrix Factorization) menggunakan beberapa parameter penting. **n_factors=100** menentukan jumlah faktor laten yang akan digunakan dalam dekomposisi matriks pengguna-item, di mana 100 faktor laten diharapkan dapat menggambarkan hubungan tersembunyi antara pengguna dan item. **n_epochs=50** mengatur jumlah iterasi pelatihan, dengan 50 epoch berarti model akan melakukan pembaruan sebanyak 50 kali untuk mencari solusi optimal. Sedangkan **reg_pu=0.05** dan **reg_qi=0.05** adalah parameter regularisasi yang digunakan untuk mencegah overfitting. **reg_pu** memberikan penalti pada faktor laten pengguna, sementara reg_qi memberi penalti pada faktor laten item, keduanya dengan nilai **0.05** untuk menjaga model agar tetap generalisasi dan tidak terlalu menyesuaikan diri dengan data latihan.

- **Co Clustering**
    Co-Clustering adalah metode dalam sistem rekomendasi yang melakukan clustering pada pengguna dan item secara bersamaan. Alih-alih hanya memfokuskan pada satu dimensi seperti pengguna atau item, Co-Clustering membagi matriks pengguna-item menjadi sub-matriks yang lebih kecil di mana pengguna dan item dalam setiap cluster memiliki preferensi atau karakteristik serupa. Metode ini sangat efektif pada data sparse, di mana tidak semua pengguna memberikan rating untuk setiap item, dan membantu memberikan rekomendasi yang lebih relevan dengan mengidentifikasi pola tersembunyi antara kelompok pengguna dan item. Sama halnya dengan NMF, untuk menentukan parameter terbaiknya menggunakan GridSearchCV. Sehingga implementasinya sebagai berikut:
    `model_co_clustering = CoClustering(n_cltr_u=1, n_cltr_i=1, n_epochs=10)`
    
    Pada kode tersebut, **`n_cltr_u=1`** menentukan jumlah cluster untuk pengguna, yang berarti hanya ada satu cluster untuk seluruh pengguna dalam model Co-Clustering. Demikian juga, **`n_cltr_i=1`** menetapkan bahwa hanya ada satu cluster untuk item, sehingga item tidak dikelompokkan lebih lanjut. Sementara itu, **`n_epochs=10`** mengatur jumlah iterasi atau epoch yang dilakukan selama pelatihan model, dengan 10 epoch berarti model akan melakukan pembaruan sebanyak 10 kali untuk mencapai hasil clustering yang optimal.

### Top 10 Rekomendasi untuk wisatawan dengan User_id = 211
- Top 5 wisatawan memberikan rating tertinggi
    | Place_Id | Place_Name   | Category    | 
    |----------|-------------|--------------|
    |202 | Pantai Ngandong | Bahari|
    |31 | Wisata Alam Mangrove Angke | Cagar Alam|
    |37 | Bumi Perkemahan Cibubur | Taman Hiburan|
    |49 | Galeri Indonesia Kaya | Budaya|
    |125 | Alun-alun Utara Keraton Yogyakarta | Budaya|

- Top 10 Rekomendasi dari model Slope One
    | Place_Id | Place_Name   | Category    | 
    |----------|-------------|--------------|
    |138 | Jogja Exotarium | Taman Hiburan|
    |52 | Kampung Cina |Budaya|
    |253 | Selasar Sunaryo Art Space | Taman Hiburan|
    |160 | Pasar Kebon Empring Bintaran | Pusat Perbelanjaan|
    |254 | Teras Cikapundung BBWS | Taman Hiburan|
    |189 | Pantai Nguluran | Bahari|
    |32 | Setu Babakan | Budaya|
    |392 | Ekowisata Mangrove Wonorejo | Cagar Alam|
    |51 | Jakarta Planetarium |Taman Hiburan|
    |237 | Panghegar Waterboom Bandung | Taman Hiburan|

- Top 10 Rekomendasi dari model NMF
    | Place_Id | Place_Name   | Category    | 
    |----------|-------------|--------------|
    |401 | Taman Keputran | Taman Hiburan|
    |51 | Jakarta Planetarium | Taman Hiburan|
    |416 | Keraton Surabaya | Budaya|
    |134 | Desa Wisata Gamplong | Taman Hiburan|
    |112 | Bukit Bintang Yogyakarta | Taman Hiburan|
    |433 | Museum Mpu Tantular | Budaya|
    |367 | Wisata Lereng Kelir | Cagar Alam|
    |28 | Wisata Agro Edukatif Istana Susu Cibugary | Taman Hiburan|
    |254 | Teras Cikapundung BBWS | Taman Hiburan|
    |132 | Air Terjun Kedung Pedut | Cagar Alam|

- Top 10 Rekomendasi dari model Co Clustering
    | Place_Id | Place_Name   | Category    | 
    |----------|-------------|--------------|
    |254 | Teras Cikapundung BBWS | Taman Hiburan|
    |416 | Keraton Surabaya | Budaya|
    |52 | Kampung Cina | Budaya|
    |28 | Wisata Agro Edukatif Istana Susu Cibugary | Taman Hiburan|
    |401 | Taman Keputran | Taman Hiburan|
    |183 | Jogja Bay Pirates Adventure Waterpark | Taman Hiburan|
    |115 | Monumen Sanapati | Budaya|
    |157 | Pantai Baron |Bahari|
    |112 | Bukit Bintang Yogyakarta | Taman Hiburan|
    |97 |Monumen Yogya Kembali | Budaya|

### Kelebihan dan Kekurangan dari Algoritma yang digunakan
**Slope One** adalah algoritma yang sederhana dan cepat, cocok untuk dataset kecil dengan sedikit data, namun kurang efektif pada dataset besar dan kompleks. **NMF (Non-negative Matrix Factorization)** mampu menangkap pola laten yang lebih dalam dan efektif untuk data sparse, namun memerlukan waktu pelatihan yang lebih lama dan sangat bergantung pada parameter regularisasi. **Co-Clustering** mengelompokkan pengguna dan item secara bersamaan, efektif pada dataset sparse dan memberikan rekomendasi yang lebih relevan, namun membutuhkan lebih banyak sumber daya dan sulit diinterpretasikan. Secara keseluruhan, Slope One lebih cocok untuk sistem sederhana, NMF untuk menangani data kompleks, dan Co-Clustering memberikan hasil terbaik pada data besar dan tersembunyi.

## Evaluation
Metrik yang digunakan pada proyek ini adalah RMSE, MAE, Precision@10 dan Recall@10. **RMSE (Root Mean Squared Error)** mengukur rata-rata kesalahan kuadrat antara rating yang diprediksi dan aktual, dengan memberikan bobot lebih besar pada kesalahan besar. **MAE (Mean Absolute Error)** mengukur rata-rata selisih absolut antara prediksi dan nilai aktual, memberikan gambaran kesalahan prediksi secara keseluruhan. **Precision@10** mengukur seberapa banyak item relevan yang ada dalam 10 rekomendasi teratas, sementara **Recall@10** mengukur seberapa banyak item relevan ditemukan dari seluruh item relevan yang ada dalam 10 rekomendasi teratas. Precision lebih mengutamakan akurasi rekomendasi, sedangkan Recall lebih mengutamakan jumlah item relevan yang berhasil direkomendasikan.

Adapun hasil dari metrik evaluasinya adalah sebagai berikut:
| Algoritma       | Test RMSE | Test MAE  | Precision@10 | Recall@10 |
|-----------------|-----------|-----------|--------------|-----------|
| Slope One      | 1.531999  | 1.285102  | 0.366667     | 0.892593  |
| NMF            | 1.514674  | 1.2792    | 0.369333     | 0.901082  |
| Co-Clustering  | 1.4394    | 1.230725  | 0.367667     | 0.893437  |
Berdasarkan hasil evaluasi, meskipun NMF (Non-negative Matrix Factorization) unggul dalam RMSE (1.514674) dan MAE (1.2792), Co-Clustering menunjukkan kinerja terbaik dalam Recall@10 dengan nilai 0.893437, sedikit lebih tinggi dari NMF yang memiliki 0.901082. Dalam Precision@10, NMF sedikit lebih unggul dengan nilai 0.369333, sedangkan Co-Clustering memperoleh 0.367667. Namun,  tujuan utama dalam sistem rekomendasi adalah memberikan rekomendasi yang **lebih relevan** dalam 10 rekomendasi teratas, maka dipastikan **Co-Clustering** adalah algoritma yang lebih optimal, karena memiliki nilai Recall@10 yang lebih tinggi. Sehingga, Co-Clustering dapat dianggap sebagai algoritma yang lebih baik dalam konteks ini.

## Referensi
- [1] https://kemenparekraf.go.id/direktori-statistik/statistik-kunjungan-wisatawan-mancanegara-bulan-januari-2024
- [2] https://surprise.readthedocs.io/en/stable/index.html