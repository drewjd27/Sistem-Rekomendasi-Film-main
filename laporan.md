# Laporan Proyek Machine Learning - Sistem Rekomendasi Film 

Nama : Andrew Jonatan Damanik

---
## Project Overview

Sistem rekomendasi film menjadi sangat penting di era digital saat ini, di mana pengguna dihadapkan pada ribuan pilihan film di berbagai platform streaming. Tanpa sistem rekomendasi yang baik, pengguna akan kesulitan menemukan film yang sesuai dengan preferensi mereka, sehingga dapat menurunkan pengalaman pengguna dan keterlibatan di platform. Sistem rekomendasi yang efektif mampu mempelajari pola tontonan, memberikan prediksi yang akurat, dan menyesuaikan rekomendasi secara dinamis sesuai perubahan preferensi pengguna.

Sistem rekomendasi telah terbukti meningkatkan loyalitas pengguna dan meningkatkan waktu yang dihabiskan di platform [1]. Oleh karena itu, membangun sistem rekomendasi yang relevan dan akurat sangat penting untuk meningkatkan kepuasan pengguna dan daya saing platform streaming.

## Business Understanding

### Problem Statements

- Bagaimana membangun sistem rekomendasi film yang dapat memberikan rekomendasi relevan sesuai preferensi pengguna?
- Bagaimana mengukur dan membandingkan efektivitas dua pendekatan sistem rekomendasi, yaitu Content-Based Filtering dan Collaborative Filtering?

### Goals

- Mengembangkan sistem rekomendasi film yang mampu memberikan rekomendasi personal kepada pengguna.
- Membandingkan dua pendekatan utama (Content-Based Filtering dan Collaborative Filtering) untuk mengetahui kelebihan dan kekurangannya dalam konteks dataset yang digunakan.

### Solution Approach

Untuk mencapai tujuan di atas, digunakan dua pendekatan utama:

1. **Content-Based Filtering**  
   Sistem merekomendasikan film berdasarkan kemiripan konten (genre) antar film. Pendekatan ini cocok jika informasi konten film tersedia lengkap dan preferensi pengguna dapat diwakili oleh fitur film.

2. **Collaborative Filtering**  
   Sistem merekomendasikan film berdasarkan pola interaksi pengguna lain yang memiliki preferensi serupa. Pendekatan ini efektif jika terdapat banyak data interaksi (rating) antar pengguna dan film.

Kedua pendekatan akan diimplementasikan dan dievaluasi untuk mengetahui keunggulan dan keterbatasannya.

## Data Understanding

Dataset yang digunakan berasal dari [Kaggle: The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) [2] di URL https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset.
Dua file utama yang digunakan:
- `movies_metadata.csv` (informasi film)
- `ratings_small.csv` (rating pengguna terhadap film)

### Jumlah dan Kondisi Data

- **movies_metadata.csv**: 45.466 baris, 24 kolom (film)
- **ratings_small.csv**: 100.003 baris, 4 kolom (rating)
- **data duplikat, dan missing values**: Kedua dataset ini masih memiliki data duplikat dan missing values. Ini akan diperlihatkan pencariannya dan penanganannya pada tahap Data Preparation di laporan ini.

### Variabel pada Dataset

#### movies_metadata.csv

| Variabel                | Keterangan                                                                 | Tipe Data |
|-------------------------|-----------------------------------------------------------------------------|-----------|
| adult                   | Apakah film mengandung konten dewasa                                       | object    |
| belongs_to_collection   | Koleksi film/franchise                                                     | object    |
| budget                  | Anggaran produksi                                                          | object    |
| genres                  | Daftar genre (list of dict)                                                | object    |
| homepage                | Situs web resmi film                                                       | object    |
| id                      | ID unik film                                                               | object    |
| imdb_id                 | ID film di IMDb                                                            | object    |
| original_language       | Bahasa asli film                                                           | object    |
| original_title          | Judul asli                                                                 | object    |
| overview                | Sinopsis                                                                   | object    |
| popularity              | Skor popularitas                                                           | object    |
| poster_path             | Path poster                                                                | object    |
| production_companies    | Studio produksi                                                            | object    |
| production_countries    | Negara produksi                                                            | object    |
| release_date            | Tanggal rilis                                                              | object    |
| revenue                 | Pendapatan kotor                                                           | float64   |
| runtime                 | Durasi (menit)                                                             | float64   |
| spoken_languages        | Bahasa yang digunakan                                                      | object    |
| status                  | Status distribusi                                                          | object    |
| tagline                 | Slogan film                                                                | object    |
| title                   | Judul utama                                                                | object    |
| video                   | Apakah ada video tambahan                                                  | object    |
| vote_average            | Rata-rata rating pengguna                                                  | float64   |
| vote_count              | Jumlah rating                                                              | float64   |

#### ratings_small.csv

| Variabel   | Keterangan                                 | Tipe Data |
|------------|---------------------------------------------|-----------|
| userId     | ID unik pengguna                           | int64     |
| movieId    | ID unik film                               | int64     |
| rating     | Rating pengguna (skala 0.5-5)              | float64   |
| timestamp  | Waktu rating diberikan (UNIX time)         | int64     |

### Exploratory Data Analysis (EDA) & Insight

- **Distribusi Rating:**

   | ![image](https://github.com/user-attachments/assets/256ac2ef-f84e-4629-bbd7-e5c952deb0e2) | 
   |:--:| 
   | *Distribusi Rating Film* |

  Rating 4.0 mendominasi (28.75%), diikuti rating 3.0 (20.06%) dan 5.0 (15.09%). Rating rendah (<2.0) sangat jarang diberikan.
  
- **Distribusi Genre:**

   | ![image](https://github.com/user-attachments/assets/b1cf6b5d-a799-456a-a9c3-d5991e402759) | 
   |:--:| 
   | *Distribusi Genre Film* |

  Genre terbanyak adalah Drama, Comedy, Thriller, Romance, dan Action.
  
- **Analisis Rating Film:**

   | ![image](https://github.com/user-attachments/assets/5432b643-4da5-4b8f-8a09-51f7ebd55a80) | 
   |:--:| 
   | *Distribusi Genre Film* |

  Film dengan jumlah vote terbanyak belum tentu memiliki rating tertinggi. Terdapat perbedaan antara rating pengguna dan rating TMDB/IMDB pada beberapa film.

## Data Preparation

### Langkah-langkah Data Preparation

1. **Data Preparation Umum**
Pada tahapan Data Preparation Umum, dilakukan proses pemilihan fitur, penyesuaian nama dan tipe data, mengubah format fitur genre, menggabungkan dataframe movies dan ratings, mengatasi missing values, dan mengatasi data duplikat.
1.1. **Pemilihan Fitur Relevan:**  
Karena dataset `movies_metadata.csv` dan `ratings_small.csv` memiliki banyak fitur/ variabel, maka hanya memilih fitur yang relevan untuk membangun sistem rekomendasi. Adapun fitur yang  dipilih diuraikan sebagai berikut:
   - Dari `movies_metadata.csv`: id, genres, title  
   - Dari `ratings_small.csv`: userId, movieId, rating

1.2. **Penyesuaian Nama dan Tipe Data:**  
   - Kolom `id` yang ada di `movies_metadata.csv` diubah ke `movieId`, dan tipe data diseragamkan ke int64 agar bisa digabung dengan rating.

1.3. **Mengubah Format Fitur Genre:**  
   - Kolom `genres` diubah dari list of dict menjadi list of genre names (string), agar model rekomendasi dapat memrposes genre dari dataset.

1.4. **Menggabungkan dataframe *movies* dan *ratings*:**  
   - Data film dan rating digabung berdasarkan `movieId`. Ini bertujuan untuk dataset film/ movie memiliki variabel yang terdapat pada dataset rating, sehingga dapat dimodelkan untuk sistem rekomendasi.

1.5. **Mengatasi Missing Values**
Ini bertujuan agar data bersih, sehingga model dapat dilatih dengan baik. Karena missing value tergolong sedikit, 202 dari 44994, maka baris dengan missing value dihapus.

1.6. **Mengatasi Data Duplikat**
Penanganan bertujuan agar data bersih, sehingga model dapat dilatih dengan baik. Sama halnya dengan penanganan missing value, data duplikat berdasarkan `movieId` dihapus.

2. **Data Preparation untuk Content-Based Filtering:** 

Pada tahap ini dilakukan persiapan data sebelum dilanjutkan ke modeling content-based filtering.

2.1. **Pemilihan Fitur yang Relevan:**

Karena Content-Based Filtering berfokus pada karakteristik item film, fitur-fitur yang relevan adalah informasi yang mendeskripsikan konten film itu sendiri, seperti `movieId`, `title`, dan `genres` yang dapat mewakili isi dari film.

2.2. **Ekstraksi Fitur Teks dari Kolom Genre:**

Pada tahap ini digunakan teknik TF-IDF (Term Frequency - Inverse Document Frequency) untuk mengubah isi kolom genres menjadi representasi numerik (matriks) berbasis teks. Representasi numerik ini memungkinkan sistem mengenali kemiripan antar film berdasarkan informasi genre. Sebelum menerapkan `TfidfVectorizer`, data pada kolom genres perlu dikonversi menjadi string, karena `TfidfVectorizer` hanya dapat memproses input dalam bentuk teks.

3. **Persiapan untuk Collaborative Filtering:**  

Pada tahap ini dilakukan persiapan data sebelum dilanjutkan ke modeling collaborative filtering.

3.1. **Pemilihan Fitur yang Relevan:**

Collaborative Filtering berfokus pada pola interaksi pengguna terhadap item, maka fitur yang dipilih adalah `userId`, `movieId`, dan `rating` yang menggambarkan hubungan antara pengguna dan item.

3.2. **Melakukan Encoding UserId dan movieId:**

Label encoding pada `userId` dan `movieId` agar model dapat memproses data yang berupa string menjadi numerik.

3.3. **Normalisasi Data:**

Karena Collaborative Filtering berfokus pada pola interaksi antara pengguna dan item, maka harus dilakukan normalisasi data sebelum melatih model, agar membantu menyamakan skala nilai rating yang diberikan oleh pengguna. Sehingga model deep learning dapat lebih efektif dalam mendeteksi pola. Fitur rating yang memiliki rentang 0 hingga 5 dinormalisasi menjadi skala 0 hingga 1. Proses ini bertujuan supaya model dapat memproses input numerik secara lebih cepat dan efisien saat proses pelatihan model.

3.4. **Train-Test-Validation Data Split:**

Split data menjadi train, validation, dan test (80:10:10). Agar model dapat dilatih dengan dataset latih (train), divalidasi dengan data yang terpisah atau belum pernah dilihat saat pelatihan dengan data train dari dataset validasi (validation), dan diuji dengan data yang belum pernah dilihat juga menggunakan dataset uji (test). Ini memastikan model benar benar memiliki kemampuan prediksi yang baik pada evaluasi nanti.

**Alasan Tahapan:**  
Setiap tahapan bertujuan memastikan data bersih, konsisten, dan siap digunakan baik untuk model berbasis konten maupun interaksi pengguna.

## Modeling and Result

### 1. Content-Based Filtering

#### a. Proses

- Menghitung cosine similarity antar film berdasarkan genre.
- Membuat fungsi rekomendasi yang mengambil top-N film paling mirip berdasarkan input judul film.

#### b. Contoh Kode

```python
from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)
cos_df = pd.DataFrame(cos_sim, columns=data['title'], index=data['title'])

def film_recommendations(nama_film, similarity_data, items, top_n=10):
    # ...fungsi rekomendasi seperti pada notebook...
    return similar_movies_df

title_based_recom = film_recommendations("The Matrix", similarity_data=cos_sim, items=data[['title', 'genres']], top_n=10)
```

#### c. Hasil

- Sistem mampu memberikan rekomendasi film dengan genre serupa.
- Contoh output: 10 film dengan genre mirip "The Matrix" (Action, Science Fiction) dan skor similarity tinggi.

```
🎬 Rekomendasi film mirip dengan 'The Matrix':

1. Fallout | Genre: Action Science Fiction | Similarity: 1.00
2. Escape from New York | Genre: Science Fiction Action | Similarity: 1.00
3. The Demolitionist | Genre: Action Science Fiction | Similarity: 1.00
4. Interceptor Force 2 | Genre: Action Science Fiction | Similarity: 1.00
5. Escape from the Planet of the Apes | Genre: Action Science Fiction | Similarity: 1.00
6. The Chronicles of Riddick | Genre: Action Science Fiction | Similarity: 1.00
7. Conquest of the Planet of the Apes | Genre: Action Science Fiction | Similarity: 1.00
8. The Matrix | Genre: Action Science Fiction | Similarity: 1.00
9. I, Robot | Genre: Action Science Fiction | Similarity: 1.00
10. The Running Man | Genre: Action Science Fiction | Similarity: 1.00
```
#### d. Kelebihan & Kekurangan

- **Kelebihan:** Tidak membutuhkan data interaksi pengguna, cocok untuk pengguna baru.
- **Kekurangan:** Rekomendasi terbatas pada kemiripan konten, tidak bisa menangkap pola preferensi unik pengguna.

---

### 2. Collaborative Filtering

#### a. Proses

  | ![Screenshot (144)](https://github.com/user-attachments/assets/1e1932ac-bc84-4610-afb8-3de902aa2a05) | 
  |:--:| 
  | *Arsitektur Model Deep Learning untuk Collaborative Filtering* |

- Mendefinisikan arsitektur model deep learning untuk mempelajari interaksi user-movie. 
    - Pada arsitektur model dibuatlah embedding terpisah untuk pengguna dan film. Vektor embedding ini akan dipelajari selama proses pelatihan untuk menangkap preferensi pengguna dan karakteristik film. 
    - Model ini juga mencakup bias untuk setiap pengguna dan film. Bias ini menangkap kecenderungan umum user dan film. Dengan adanya bias, model bisa lebih fleksibel dalam menangani kecenderungan pengguna dan film.
    - Setelah embedding dibuat untuk setiap user dan movie, dilakukan operasi dot product antara vektor embedding user dan movie. Hasil dot product ini menunjukkan seberapa besar kecocokan antara pengguna dan film yang bersangkutan.
    - Bias User dan Movie ditambahkan ke hasil perkalian dot product untuk memberikan skor kecocokan yang lebih akurat.
    - Setelah hasil dot product dan bias ditambahkan, digunakan fungsi aktivasi sigmoid untuk membatasi output antara 0 dan 1, sehingga bisa disesuaikan dengan rating yang dinormalisasi.
    - Output akhir dari arsitektur model ini adalah prediksi rating untuk pasangan user dan movie tertentu. Prediksi ini berada dalam skala 0 hingga 1, karena rating telah dinormalisasi sebelumnya.
- Kompilasi Model
    - Loss Function: Mean Squared Error (MSE) – untuk menghitung selisih antara rating yang diprediksi dan rating asli yang sudah dinormalisasi.
    - Optimizer: Adam – dengan learning rate default.
    - Evaluation Metrics menggunakan Mean Absolute Error (MAE), dan Root Mean Squared Error (RMSE).
- Training Model. Setelah arsitektur model telah didefinisikan, dan model sudah di compile, maka model akan dilatih dengan dataset training. Adapun `batch size` yang digunakan adalah 32, dan `epoch` nya adalah 15.
- Input: user dan movie yang sudah di-encode.
- Output: prediksi rating (skala 0-1). 
- Fungsi rekomendasi: memberikan top-N film dengan prediksi rating tertinggi untuk user tertentu.

#### b. Contoh Kode

```python
def recommend_movies_for_user(user_id, model, data_cf, movie_df, top_n=10):
    # ...fungsi rekomendasi collaborative filtering seperti pada notebook...
    return top_rated_by_user, recommendations

top5_rated, top10_recommendations = recommend_movies_for_user(
    user_id=1,
    model=model,
    data_cf=preparation_cf,
    movie_df=df_movies,
    top_n=10
)
```

#### c. Hasil

- Sistem mampu memberikan rekomendasi personal berdasarkan pola rating pengguna lain.
- Output: 10 film dengan prediksi rating tertinggi untuk user tertentu.

Top 5 Film dengan Rating Tertinggi oleh User : 1

| No | Title                          | Genres                                   | Rating |
|----|--------------------------------|------------------------------------------|--------|
| 0  | Beverly Hills Cop             | [Action, Comedy, Crime]                  | 5.0    |
| 1  | Who's Afraid of Virginia Woolf? | [Drama]                                 | 5.0    |
| 2  | AVP: Alien vs. Predator       | [Adventure, Science Fiction, Action]     | 5.0    |
| 3  | The Secret Life of Words      | [Drama, Romance]                         | 5.0    |


Top 10 Film Rekomendasi untuk User : 1

| No | Title                     | Genres                                                     | Predicted Rating |
|----|---------------------------|-------------------------------------------------------------|------------------|
| 1  | A Fistful of Dollars      | [Western]                                                   | 0.945035         |
| 2  | Requiem                   | [Action, Horror, Thriller]                                  | 0.808208         |
| 3  | Journey to Italy          | [Romance, Drama]                                            | 0.799308         |
| 4  | Magnetic Rose             | [Animation, Science Fiction]                                | 0.794201         |
| 5  | Ghost World               | [Comedy, Drama]                                             | 0.790034         |
| 6  | The Elementary Particles  | [Drama, Romance]                                            | 0.789681         |
| 7  | The Ewok Adventure        | [Adventure, Family, Fantasy, Science Fiction]              | 0.789060         |
| 8  | The Patriot               | [Drama, History, War, Action]                               | 0.782098         |
| 9  | Tanguy                    | [Comedy]                                                    | 0.780484         |
| 10 | Star Trek: Insurrection   | [Science Fiction, Action, Adventure, Thriller]              | 0.779822         |



#### d. Kelebihan & Kekurangan

- **Kelebihan:** Dapat menangkap pola preferensi unik pengguna, rekomendasi lebih personal.
- **Kekurangan:** Membutuhkan data interaksi yang cukup banyak, cold start problem untuk user/film baru yang belum memiliki interaksi historis.

## Evaluation

### Content-Based Filtering

- **Metrik:** Precision dan Recall
- Precision adalah metrik yang mengukur proporsi prediksi positif yang benar-benar relevan atau benar. Dalam konteks klasifikasi, precision dihitung sebagai jumlah true positive dibagi dengan jumlah total prediksi positif (true positive + false positive) [3]
- Recall diartikan sebagai proporsi item relevan yang berhasil ditemukan oleh model dari seluruh item relevan yang ada di data [3].
- **Formula:**
  - Precision = (Jumlah rekomendasi relevan) / (Total rekomendasi yang diberikan)
  - Recall = (Jumlah rekomendasi relevan) / (Total item relevan di data)
- **Hasil:**

  | ![Screenshot (144)](https://github.com/user-attachments/assets/5b0543b0-52fe-4e34-b978-1a3cab167c70) | 
  |:--:| 
  | *Evaluasi MAE dan RMSE pada Test Dataset* |

  Precision: 100% (semua rekomendasi relevan dengan preferensi genre)  
  Recall: 100% (semua film relevan berhasil direkomendasikan)

### Collaborative Filtering

- **Metrik:** Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE)
- MAE mengukur rata-rata selisih absolut antara nilai prediksi model dan nilai aktual, sehingga memberikan gambaran seberapa besar kesalahan prediksi secara rata-rata dalam satuan yang sama dengan data aslinya [4].
- Root Mean Squared Error (RMSE) adalah salah satu metrik evaluasi yang paling umum digunakan dalam machine learning, khususnya untuk mengukur seberapa akurat model dalam memprediksi nilai numerik. RMSE memberikan gambaran seberapa besar rata-rata kesalahan prediksi model terhadap nilai sebenarnya, dengan memberikan penalti lebih besar pada kesalahan yang lebih besar [5].
- **Formula:**

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_{\text{true},i} - y_{\text{pred},i} \right|
$$

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_{\text{true},i} - y_{\text{pred},i} \right)^2 }
$$
- **Hasil:**

```
Evaluasi MAE pada test dataset: 0.20269782841205597
Evaluasi RMSE pada test dataset: 0.2480652779340744
```

  | ![Screenshot (144)](https://github.com/user-attachments/assets/fada832e-90a9-4960-bbaa-761c46c9c37c) | 
  |:--:| 
  | *Evaluasi MAE dan RMSE pada Test Dataset* |
 
  - MAE dan RMSE pada test dataset menunjukkan model cukup baik dalam memprediksi rating pengguna.
  - Grafik MAE dan RMSE selama training menunjukkan model stabil dan tidak overfitting.

---

## Daftar Pustaka

[1] A. K. Gupta, “Real-World Evaluation: Hybrid Recommender System and User Engagement,” pp. 1–6, Jun. 2024, doi: https://doi.org/10.1109/apci61480.2024.10617191.

[2] R. Banik, “The Movies Dataset,” www.kaggle.com. https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

‌[3] B. Juba and H. S. Le, “Precision-Recall versus Accuracy and the Role of Large Data Sets,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 1, pp. 4039–4048, Jul. 2019, doi: https://doi.org/10.1609/aaai.v33i01.33014039.

‌[4] J. Qi, J. Du, S. M. Siniscalchi, X. Ma, and C.-H. Lee, “On Mean Absolute Error for Deep Neural Network Based Vector-to-Vector Regression,” IEEE Signal Processing Letters, vol. 27, pp. 1485–1489, 2020, doi: https://doi.org/10.1109/lsp.2020.3016837.

‌[5] A. S. B. Karno, “Prediksi Data Time Series Saham Bank BRI Dengan Mesin Belajar LSTM (Long ShortTerm Memory),” Journal of Informatic and Information Security, vol. 1, no. 1, pp. 1–8, May 2020, doi: https://doi.org/10.31599/jiforty.v1i1.133.
