# %% [markdown]
# # Sistem Rekomendasi Film | Collaborative & Content-Based Filtering

# %% [markdown]
# ## A. Business Understanding
# Dalam menghadapi tantangan banyaknya pilihan film yang tersedia, sistem rekomendasi terus berkembang dengan mengadopsi berbagai pendekatan teknologi, seperti machine learning dan analisis data perilaku pengguna. Pendekatan ini memungkinkan sistem untuk mempelajari pola tontonan, memberi prediksi yang lebih akurat, dan menyesuaikan rekomendasi secara dinamis. Tidak hanya itu, sistem yang cerdas juga dapat mengenali perubahan preferensi pengguna seiring waktu, sehingga rekomendasi yang diberikan tetap relevan. Dengan demikian, sistem rekomendasi tidak hanya meningkatkan pengalaman pengguna, tetapi juga membantu platform streaming mempertahankan loyalitas pengguna dan meningkatkan keterlibatan mereka.

# %% [markdown]
# ## B. Data Understanting

# %% [markdown]
# ## 1. Import Library

# %%
import ast
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')

# Library Visualisasi dan analisis Data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# %matplotlib inline  # Removed for script compatibility

# Library deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Library pemrosesan dan metrik
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## 2. Data Loading

# %%
# Menyimpan data dalam variabel
movies = pd.read_csv("dataset/movies_metadata.csv")
credits = pd.read_csv("dataset/credits.csv")
keywords = pd.read_csv("dataset/keywords.csv")
ratings = pd.read_csv("dataset/ratings.csv")
ratings_small = pd.read_csv("dataset/ratings_small.csv")
links_small = pd.read_csv("dataset/links_small.csv")
links = pd.read_csv("dataset/links.csv")

# %%
# movies
print(f'Jumlah data movies adalah --> {movies["id"].nunique()}')

print('_'*75)

# credits
print(f'Jumlah data rating keseluruhan --> {len(ratings)}')
print(f'Jumlah pengguna yang memberikan rating ke film --> {ratings["userId"].nunique()}')
print(f'Jumlah film yang memiliki rating --> {ratings["movieId"].nunique()}')

print('_'*75)

# ratings_small
print(f'Jumlah data rating_small keseluruhan --> {len(ratings_small)}')
print(f'Jumlah pengguna yang memberikan rating ke film --> {ratings_small["userId"].nunique()}')
print(f'Jumlah film yang memiliki rating --> {ratings_small["movieId"].nunique()}')

print('_'*75)

# links
print(f'Jumlah data links dari imdb dan tmdb dari masing-masing movie   --> {len(links)}')
# links_small
print(f'Jumlah data links_small dari imdb dan tmdb dari masing-masing movie  --> {len(links_small)}')

print('_'*75)

# credits
print(f'Jumlah data pemain dan kru --> {len(credits)}')
print(f'Jumlah data keywords --> {len(keywords)}')

# %% [markdown]
# Dalam penelitian ini, saya hanya akan menggunakan dua dataset, yaitu **movies_metadata.csv** dan **ratings_small.csv**, karena kedua dataset ini cukup untuk membangun sistem rekomendasi.

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %% [markdown]
# Tujuan tahap Exploratory Data Analysis (EDA) untuk menganalisis distribusi film, rating, serta memahami hubungan antar fitur dalam dataset.

# %% [markdown]
# ### 3.1. Deskripsi Variabel

# %% [markdown]
# #### 3.1.1. File Movies

# %%
movies.head()

# %%
movies.info()

# %% [markdown]
# Berdasarkan output di atas, variabel `movies` terdiri dari **45.466 baris** dan **24 kolom**. Adapun deskripsi fitur-fiturnya dapat dilihat sebagai berikut:
# 
# | **Variabel**                | **Keterangan** | **Tipe Data** |
# |----------------------------|----------------|---------------|
# | `adult`                    | Menandakan apakah film mengandung konten dewasa. Biasanya bernilai `True` atau `False`. | object        |
# | `belongs_to_collection`    | Menunjukkan apakah film termasuk dalam suatu seri atau koleksi tertentu, seperti franchise film. Biasanya berbentuk string atau format JSON. | object        |
# | `budget`                   | Total anggaran produksi film, umumnya dalam satuan dolar AS (USD). | object        |
# | `genres`                   | Daftar genre yang dimiliki film, misalnya Action, Drama, atau Comedy. Tersedia dalam bentuk list atau JSON. | object        |
# | `homepage`                 | Tautan ke situs web resmi film. | object        |
# | `id`                       | ID unik film di dalam sistem database (seperti TMDb). | object        |
# | `imdb_id`                  | ID khusus film pada database IMDb. | object        |
# | `original_language`        | Kode bahasa asli film, mengikuti format ISO 639-1 (contoh: `en` untuk Bahasa Inggris). | object        |
# | `original_title`           | Judul asli film sesuai dengan versi bahasa produksinya. | object        |
# | `overview`                 | Ringkasan atau sinopsis singkat mengenai cerita film. | object        |
# | `popularity`               | Skor popularitas film yang dihitung berdasarkan sistem tertentu dari platform. | object        |
# | `poster_path`              | Path atau lokasi file gambar poster film, biasanya digunakan bersama URL dasar untuk mengakses gambar. | object        |
# | `production_companies`     | Informasi tentang studio atau perusahaan yang memproduksi film, biasanya berupa daftar dalam format JSON. | object        |
# | `production_countries`     | Negara tempat film diproduksi, tersedia dalam format JSON yang berisi nama dan kode negara. | object        |
# | `release_date`             | Tanggal film dirilis, dengan format `YYYY-MM-DD`. | object        |
# | `revenue`                  | Pendapatan kotor yang dihasilkan film, biasanya dalam USD. | float64       |
# | `runtime`                  | Durasi total film dalam satuan menit. | float64       |
# | `spoken_languages`         | Bahasa yang digunakan dalam percakapan film, dicatat dalam format JSON. | object        |
# | `status`                   | Status distribusi film, seperti `Released` atau `In Production`. | object        |
# | `tagline`                  | Slogan atau kutipan promosi yang terkait dengan film. | object        |
# | `title`                    | Judul utama film yang umum digunakan untuk distribusi atau promosi. | object        |
# | `video`                    | Menunjukkan apakah film memiliki video tambahan terkait. Nilainya berupa `True` atau `False`. | object        |
# | `vote_average`             | Rata-rata skor atau penilaian yang diberikan oleh pengguna (misalnya IMDb atau TMDb) terhadap film, biasanya dalam skala 1–10. | float64       |
# | `vote_count`               | Jumlah total suara atau penilaian yang diterima oleh film. | float64       |
# 

# %% [markdown]
# #### 3.1.2. File Ratings

# %%
ratings_small.head()

# %%
ratings_small.info()

# %% [markdown]
# Berdasarkan output di atas, variabel `ratings_small` terdiri dari subset rating lengkap yaitu **100.003 baris** dan **4 kolom**. Adapun deskripsi fitur-fiturnya dapat dilihat sebagai berikut:
# 
# | **Variabel**   | **Keterangan**                                                                                                                                      | **Tipe Data** |
# |----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
# | `userId`       | Merupakan identitas unik yang diberikan kepada setiap pengguna yang memberikan rating. Digunakan untuk membedakan antar pengguna secara anonim.     | int64         |
# | `movieId`      | Kode unik yang mewakili setiap film yang dinilai oleh pengguna. ID ini dapat digunakan untuk mengakses informasi lebih lengkap tentang film tersebut.| int64         |
# | `rating`       | Skor atau nilai evaluasi yang diberikan pengguna terhadap sebuah film, biasanya dalam skala 1 sampai 5. Semakin tinggi nilainya, semakin positif penilaian pengguna. | float64       |
# | `timestamp`    | Menunjukkan kapan rating diberikan.                                                                                                                 | int64         |
# 
# Saya melihat bahwa tipe data pada fitur `id` di dataframe `movies` (object) dan fitur `movieId` di `ratings` (int64) tidak konsisten. Saya akan mengubah tipe datanya ke int64 agar seragam.

# %% [markdown]
# ### 3.2. Hitung Total Data dari Dataset

# %%
print(f'Jumlah movies adalah --> {movies["id"].nunique()}')
print(f'Jumlah rating adalah --> {len(ratings_small)}')
print(f'Jumlah pengguna yang memberikan rating adalah --> {ratings_small["userId"].nunique()}')
print(f'Jumlah film yang memiliki rating adalah --> {ratings_small["movieId"].nunique()}')

# %% [markdown]
# Dari output diatas terdapat 45.436 film pada dataset movies, 9066 film yang memiliki rating dan 671 pengguna yang memberikan rating pada film.

# %% [markdown]
# ### 3.3. Informasi Statistik Dataset 

# %%
movies.describe()

# %% [markdown]
# - Insight:
# 
#   - Vote Average menunjukkan rata-rata rating film, yang dimulai dari 0 dengan  berkisar antara 5 hingga 7, lalu nilai tertinggi 10.
# 
#   - Vote Count menunjukkan banyaknya ulasan yang diterima film. Banyak film yang memiliki jumlah rating rendah (bahkan ada film yang memiliki jumlah rating 0), namun ada beberapa film yang mendapatkan ribuan jumlah rating.

# %%
ratings_small.describe()

# %% [markdown]
# - Insight:
# 
#     - UserId menunjukkan pengguna yang memberikan rating.
# 
#     - MovieId menunjukkan film yang dinilai/memiliki rating.
# 
#     - Rating menunjukkan bahwa sebagian besar rating yang diberikan adalah 3 hingga 5, dengan rata-rata rating adalah 3.543608
# 
#     - Timestamp menunjukkan waktu ketika rating diberikan. Timestamp dicatat dalam bentuk waktu UNIX.

# %% [markdown]
# ### 3.4. Distribusi Rating Film

# %%
# Data rating dalam persen
count_rating = ratings_small.groupby('rating')['userId'].count()
count_rating_percentage = count_rating/len(ratings_small) * 100

# Barplot distribusi rating
plt.figure(figsize=(10,5))
ax = sns.barplot(x=count_rating.index, y=count_rating_percentage, color='limegreen')
for p in ax.patches:
  plt.text(p.get_x() + p.get_width()/2., p.get_height()+0.5, f'{p.get_height():.2f}%', ha='center', va='center')
plt.title('Distribusi Rating Film')
plt.xlabel('Rating')
plt.ylabel('Persentase Rating')
plt.show()

# %% [markdown]
# Insight:
# 
# **1. Rating 4.0 Mendominasi (28.75%)**
# 
# * Ini menunjukkan bahwa sebagian besar pengguna **cenderung memberi nilai tinggi tapi tidak maksimal**.
# * Kemungkinan besar pengguna bersikap realistis, tidak langsung kasih 5 kecuali filmnya sangat berkesan.
# 
# **2. Rating 3.0 Juga Cukup Populer (20.06%)**
# 
# * Rating ini bisa dianggap sebagai **rating “netral” atau cukup”**.
# * Ini menandakan banyak pengguna merasa film yang mereka tonton tidak terlalu bagus, tapi juga tidak jelek.
# 
# **3. Rating 5.0 Masih Tinggi (15.09%)**
# 
# * Cukup banyak pengguna yang memberi **rating sempurna**.
# * Ini bisa digunakan sebagai sinyal bahwa film tersebut benar-benar disukai, meskipun tetap perlu diperiksa jumlah user-nya.
# 
# **4. Rating Rendah Jarang Diberikan**
# 
# * Rating 0.5, 1.0, dan 1.5 masing-masing **kurang dari 3.5%**.
# * Ini mengindikasikan bahwa pengguna jarang memberi nilai sangat rendah.
# * Bisa jadi karena:
# 
#   * Mereka jarang menonton film yang jelek.
#   * Atau memang enggan memberi rating rendah (bias persepsi pengguna).

# %% [markdown]
# ### 3.5. Melihat Distribusi Genre dalam Film

# %%
movies.head()

# %% [markdown]
# Berdasarkan struktur kolom `genres`, datanya berupa **list of dictionaries**. Jadi, saya perlu ekstrak name dari setiap dict dalam list tersebut untuk tiap baris, lalu menghitung kemunculan masing-masing genre.

# %%
# Pastikan kolom 'genres' diparsing dengan benar
movies_genres = movies.copy()
movies_genres['genres'] = movies_genres['genres'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Ekstrak nama genre dan simpan dalam satu list panjang
all_genres = []
for genre_list in movies_genres['genres']:
    all_genres.extend([genre['name'] for genre in genre_list])

# Hitung frekuensi masing-masing genre
genre_counts = pd.Series(all_genres).value_counts().head(10)

# Tampilkan sebagai tabel
genre_table = genre_counts.reset_index()
genre_table.columns = ['Genre', 'Jumlah']
print("Top 10 Genre Film Terbanyak:")
print("=======================================")
print(genre_table)

# Visualisasi
plt.figure(figsize=(10,5))
sns.barplot(x='Jumlah', y='Genre', data=genre_table, color='limegreen')
plt.title('Top 10 Genre Film Terbanyak')
plt.xlabel('Jumlah Film')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()

# %% [markdown]
# Genre adalah fitur penting dalam content-based filtering. Adapun insight dari output di atas dapat dilihat sebagai berikut:
# 
# 1. **Dominasi Genre Drama**
# 
#    * Dengan **20.265 film**, *Drama* menjadi genre paling umum dalam dataset.
#    * Ini wajar karena drama adalah genre yang fleksibel dan sering digabungkan dengan genre lain seperti romance, crime, atau thriller.
# 
# 2. **Popularitas Genre Komedi**
# 
#    * *Comedy* berada di posisi kedua dengan **13.182 film**.
#    * Genre ini biasanya memiliki audience yang luas dan relatif mudah diproduksi dengan variasi yang banyak, baik standalone maupun campuran.
# 
# 3. **Thriller, Romance, dan Action Juga Signifikan**
# 
#    * Ketiganya masing-masing berada di angka **7.624**, **6.735**, dan **6.596**.
#    * *Thriller* dan *Action* sering diminati karena ketegangan dan visual yang menarik, sedangkan *Romance* banyak diminati karena keterkaitan emosional.
# 
# 4. **Genre-Genre Khusus Mulai dari Posisi 6 ke Bawah**
# 
#    * *Horror*, *Crime*, dan *Documentary* berada di posisi menengah ke bawah tapi tetap cukup banyak.
#    * *Science Fiction* meskipun termasuk genre populer secara global, dalam dataset ini hanya muncul **3.049 kali**, menunjukkan genre ini tidak sebanyak drama atau komedi secara keseluruhan.

# %% [markdown]
# ### 3.6. Analisis Rating dengan Film

# %%
# Copy data movie untuk analisis
movies_copy = movies.copy()

# Drop baris yang id-nya kosong
movies_copy = movies_copy.dropna(subset=['id'])

# Saring id dari movie yang hanya berupa angka
movies_copy = movies_copy[movies_copy['id'].apply(lambda x: str(x).isnumeric())]

# Konversi id movie dari object ke int
movies_copy['id'] = movies_copy['id'].astype('int')

# Ubah nama kolom id ke movieId agar bisa melakukan join dengan dataframe rating
movies_copy = movies_copy.rename(columns={'id':'movieId'})

# Join ke dataframe untuk mencari movie yang telah diberikan rating
movie_ratings = pd.merge(left=movies_copy, right=ratings_small, on='movieId', how='inner')

# Kelompokkan berdasarkan judul movie
df_movie_rating = movie_ratings.groupby('title').agg(
    user_rating_mean=('rating', 'mean'),                # Rata-rata rating dari pengguna
    tmdb_imdb_rating=('vote_average', 'first'),         # Rating dari TMDB/IMDB (satu nilai unik per film)
    count_user_voted=('userId', 'count')                      # Jumlah rating yang masuk dari pengguna
)

# Urutkan berdasarkan jumlah vote terbanyak
df_movie_rating = df_movie_rating.sort_values(by='count_user_voted', ascending=False)

# Tampilkan Top 10 film berdasarkan jumlah vote
print(df_movie_rating.head(10))


# Reset index agar kolom title bisa digunakan dalam visualisasi
top10 = df_movie_rating.head(10).reset_index()

plt.figure(figsize=(12,6))
sns.barplot(
    y='title',
    x='count_user_voted',
    data=top10,
    color='limegreen'
)

plt.title('Top 10 Film Berdasarkan Jumlah Vote Pengguna', fontsize=14)
plt.xlabel('Jumlah Vote dari Pengguna')
plt.ylabel('Judul Film')
plt.tight_layout()
plt.show()



# %% [markdown]
# Insight:
# 
# 1. **Film dengan Popularitas Tertinggi (jumlah vote terbanyak)**
# 
#    * **Terminator 3: Rise of the Machines** menempati posisi teratas dengan **324 vote** dari pengguna.
#    * Menariknya, meskipun film ini tidak mendapat rating tinggi dari TMDB (5.9), pengguna justru memberikan rating yang **relatif tinggi (4.26)**.
# 
# 2. **Perbedaan Rating Pengguna vs TMDB**
# 
#    * Ada beberapa film dengan **gap signifikan** antara rating pengguna dan rating TMDB/IMDB:
# 
#      * **The Passion of Joan of Arc** → Rating pengguna: **3.48**, tapi rating TMDB: **8.2** → menandakan film ini lebih diapresiasi secara sinematik (kritikus), tapi kurang disukai oleh pengguna biasa.
#      * **Three Colors: Red** juga mendapat rating tinggi di TMDB (**7.8**), tapi rating pengguna hanya **3.95**.
# 
# 3. **Konsistensi Rating**
# 
#    * **Solaris** punya rating tinggi dari TMDB (**7.7**) dan pengguna juga cukup tinggi (**4.13**), menunjukkan konsistensi penerimaan positif dari dua sumber.
# 

# %% [markdown]
# ## 4. Data Preparation

# %% [markdown]
# Di tahap sebelumnya dapat disimpulkan bahwa sistem rekomendasi Content-Based Filtering maupun Collaborative Filtering cocok diterapkan pada dataset ini. Pada tahapan Data Preparation, dilakukan proses pembersihan data, sehingga data bebas dari missing values dan data duplikat.

# %% [markdown]
# ### 4.1. Pemilihan Fitur yang Relevan
# Pada tahap ini, hanya beberapa fitur yang relevan atau perlu untuk membangun sistem rekomendasi yang akan diambil yaitu `id`, `genres`, dan `title` dari dataframe **movies**. Sedangkan pada dataframe **ratings**, saya akan menghapus kolom `timestamp`.

# %%
df_movies = movies[['id','genres', 'title']]
df_movies.head()

# %%
df_ratings = ratings_small.drop(columns='timestamp')
df_ratings.head()

# %% [markdown]
# ### 4.2. Penyesuaian Nama dan Tipe Data
# - Dari tahap EDA diketahui bahwa perlu dilakukan penyesuaian tipe data pada kolom id di dataframe **movies** agar konsisten dengan tipe data int64 pada kolom `movieId` di dataframe **ratings**.
# - Setelah itu, nama kolom `id` pada dataframe **movies** perlu diubah menjadi `movieId` agar dapat melakukan join antar dataframe.

# %%
# Pastikan id hanya berisi angka (numerik) dan filter nilai yang valid
df_movies = df_movies[df_movies['id'].apply(lambda x: x.isnumeric())]

# Ubah tipe data id menjadi int64
df_movies['id'] = df_movies['id'].astype('int64')

# Ganti nama kolom 'id' menjadi 'movieId'
df_movies = df_movies.rename(columns={'id': 'movieId'})

df_movies.head()

# %%
df_movies.info()

# %% [markdown]
# ### 4.3. Mengubah Format Fitur Genre
# - Mengubah format kolom genres yang berisi data dalam bentuk list of dictionaries menjadi format list of genre names agar lebih mudah dalam analisis dan pemrosesan lebih lanjut.
# - Llist kosong pada kolom genres diubah menjadi **NaN** agar baris-baris yang tidak memiliki genre dapat dengan mudah diidentifikasi dan dihapus saat diperlukan. Langkah ini membantu meminimalkan gangguan dari data yang tidak relevan dalam analisis.

# %%
# Mengubah string yang berbentuk list of dictionaries menjadi objek list menggunakan literal_eval
df_movies['genres'] = df_movies['genres'].apply(lambda x: literal_eval(x) if pd.notnull(x) else np.nan)

# Mengambil nama genre dari setiap dictionary di dalam list 'genres'.
# Jika list kosong, ubah menjadi NaN
df_movies['genres'] = df_movies['genres'].apply(lambda x: [i['name'] for i in x] if len(x) > 0 else np.nan)

df_movies.head()

# %% [markdown]
# ### 4.4. Menggabungkan dataframe *movies* dan *ratings*
# Pada tahap ini dilakukan penggabungan antara DataFrame movies dan ratings. Ini bertujuan supaya dapat menganalisa setiap film dengan rating yang diberikan oleh pengguna, sehingga informasi mengenai rating film dapat disertakan.

# %%
# Menggabungkan dataframe
df_movies_ratings = pd.merge(left=df_movies, right=df_ratings, on='movieId', how='inner')
df_movies_ratings.head()

# %% [markdown]
# ### 4.5. Mengatasi Missing Values
# Dari tahapan sebelumnya, saya melihat beberapa missing values. Hal ini terjadi karena beberapa baris mungkin memiliki data yang tidak lengkap. Pertama saya akan memeriksa dahulu missing values pada dataset.

# %%
# cek missing value
print(f'Jumlah missing value pada saat ini adalah --> {df_movies_ratings.isnull().sum().sum()}')
print(f'Jumlah data saat ini adalah --> {len(df_movies_ratings)}')

# %% [markdown]
# Karena jumlah missing values hanya sebanyak 202 data dari 44994, maka saya akan menangani missing values dengan cara menghapus baris data yang memiliki missing values.

# %%
# Hapus nilai yang missing
df_movies_ratings = df_movies_ratings.dropna()

df_cleaned = df_movies_ratings.copy()

print(f'Jumlah missing value setelah ditangani adalah --> {df_cleaned.isnull().sum().sum()}')
print(f'Jumlah data dataset setelah missing value ditangani adalah --> {len(df_cleaned)}')

# %% [markdown]
# ### 4.6. Mengatasi Data Duplikat

# %%
df_cleaned = df_cleaned.drop_duplicates(subset=['movieId'])
df_cleaned

# %% [markdown]
# ## 5. Data Preparation untuk Content-Based Filtering

# %% [markdown]
# Content-based filtering adalah metode yang digunakan dalam sistem rekomendasi yang berfokus pada karakteristik atau konten dari item-item yang ingin direkomendasikan atau dianalisis. Data yang telah dipersiapkan pada tahapan data preparation sebelumnya akan difokuskan untuk mengolah informasi dari sisi konten film untuk mengukur kemiripan antar film dan memberikan rekomendasi yang relevan.

# %%
preparation_cb = df_cleaned.copy()
preparation_cb

# %% [markdown]
# - Dari pengecekan data diatas terlihat ada 44.787 baris dan 5 kolom

# %% [markdown]
# ### 5.1. Pemilihan Fitur yang Relevan
# - Karena Content-Based Filtering berfokus pada karakteristik item film, fitur-fitur yang relevan adalah informasi yang mendeskripsikan konten film itu sendiri, seperti `movieId`, `title`, dan `genres` yang dapat mewakili isi dari film.

# %%
# pilih movieId, userId, rating
preparation_cb = preparation_cb[['movieId', 'title', 'genres']]
preparation_cb

# %% [markdown]
# ### 5.2. Ekstraksi Fitur Teks dari Kolom Genre
# Pada tahap ini saya menggunakan teknik TF-IDF (Term Frequency - Inverse Document Frequency) untuk mengubah isi kolom genres menjadi representasi numerik berbasis teks. Representasi numerik ini memungkinkan sistem mengenali kemiripan antar film berdasarkan informasi genre. Sebelum menerapkan `TfidfVectorizer`, data pada kolom genres perlu dikonversi menjadi string, karena `TfidfVectorizer` hanya dapat memproses input dalam bentuk teks.

# %%
data = preparation_cb.copy().reset_index(drop=True)

# Konversi list menjadi string
data['genres'] = data['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

# %%
# Pastikan kolom 'genres' berisi string
data['genres'] = data['genres'].apply(lambda x: ' '.join(x) if isinstance(x, (list, tuple)) else x)

# Inisiasi objek TF-IDF Vectorizer dari sklearn
vectorizer = TfidfVectorizer()

# Transformasi teks genre menjadi matriks TF-IDF
tf_idf_matrix = vectorizer.fit_transform(data['genres'])

# Menampilkan fitur/genre unik yang dihasilkan oleh TF-IDF setelah proses token
vectorizer.get_feature_names_out()

# %%
# Cek shape dari matriks tf_idfnya
tf_idf_matrix.shape

# %%
# Masukkan hasil dari matrix ke dalam dataframe

pd.DataFrame(
    tf_idf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=data['title']
)

# %% [markdown]
# ## 5. Data Preparation untuk Collaborative Filtering
# Collaborative Filtering adalah metode dalam sistem rekomendasi yang memberikan rekomendasi berdasarkan interaksi atau preferensi pengguna lain yang memiliki pola kesukaan serupa, tanpa mempertimbangkan konten atau atribut dari item tersebut. Data yang telah dipersiapkan pada tahapan data preparation awal akan difokuskan untuk mengolah informasi dari interaksi pengguna terhadap item, misalnya rating terhadap film, guna mengukur kemiripan antar pengguna atau antar fil berdasarkan pola perilaku pengguna.

# %%
preparation_cf = df_cleaned.copy()
preparation_cf

# %% [markdown]
# - Dari pengecekan data diatas terlihat ada 44.787 baris dan 5 kolom

# %% [markdown]
# ### 5.1. Pemilihan Fitur yang Relevan
# Collaborative Filtering berfokus pada pola interaksi pengguna terhadap item, maka fitur yang dipilih adalah userId, movieId, dan rating yang menggambarkan hubungan antara pengguna dan item.

# %%
preparation_cf = preparation_cf[['userId', 'movieId', 'rating']]

# urutkan datanya berdasarkan userId
preparation_cf = preparation_cf.sort_values(by='userId')
preparation_cf

# %% [markdown]
# ### 5.2. Melakukan Encoding UserId dan movieId
# Karena model deep learning hanya bisa bekerja dengan angka, maka kolom userId dan movieId harus diubah ke bentuk numerik.

# %%
# Inisiasi LabelEncoder untuk mengonversi userId dan movieId menjadi format numerik
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

# Gunakan LabelEncoder untuk mengonversi userId dan movieId ke dalam label numerik
preparation_cf['user'] = user_encoder.fit_transform(preparation_cf['userId'])
preparation_cf['movie'] = movie_encoder.fit_transform(preparation_cf['movieId'])


# Tampilkan Jumlah pengguna, Jumlah film, Minimum rating, dan Maksimum rating
print(f'Jumlah pengguna --> {preparation_cf["user"].nunique()}')
print(f'Jumlah movie --> {preparation_cf["movie"].nunique()}')
print(f'Rating minimum --> {preparation_cf["rating"].min()}')
print(f'Rating maksimum --> {preparation_cf["rating"].max()}\n')

preparation_cf.head()

# %% [markdown]
# - Data ini memiliki 671 user, 2800 film dan rating minimum 0.5 dan rating maksimum 5.0

# %% [markdown]
# ### 5.3. Normalisasi Data
# Karena Collaborative Filtering berfokus pada pola interaksi antara pengguna dan item, maka harus dilakukan normalisasi data sebelum melatih model, agar membantu menyamakan skala nilai rating yang diberikan oleh pengguna. Sehingga model deep learning dapat lebih efektif dalam mendeteksi pola. Fitur rating yang memiliki rentang 0 hingga 5 dinormalisasi menjadi skala 0 hingga 1. Proses ini bertujuan supaya model dapat memproses input numerik secara lebih cepat dan efisien saat proses pelatihan model.

# %%
# Normalisasi rating
min_rating = preparation_cf['rating'].min()
max_rating = preparation_cf['rating'].max()

preparation_cf['rating_normalized'] = (preparation_cf['rating'] - min_rating) / (max_rating - min_rating)
preparation_cf

# %% [markdown]
# ### 5.4. Train-Test-Validation Data Split
# Dalam tahap ini dilakukan proses pemisahan dataset menjadi tiga subset dengan ratio 80% untuk data pelatihan (training), 10% untuk data validasi (validation), dan 10% untuk data pengujian (testing). Pembagian ini dilakukan untuk memastikan model dapat dilatih dengan data yang cukup, diuji dengan data yang belum pernah dilihat, dan dievaluasi dengan data yang terpisah dari proses pelatihan.

# %%
# Memisahkan data menjadi 80% untuk training dan 20% untuk validasi dan testing
train_data, temp_data = train_test_split(preparation_cf, test_size=0.2, random_state=42)

# Memisahkan 20% temp_data menjadi 10% validasi dan 10% testing
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# hitung persentase data
total_data = len(preparation_cf)
train_percentage = (len(train_data) / total_data) * 100
test_percentage = (len(test_data) / total_data) * 100
validation_percentage = (len(validation_data) / total_data) * 100


# Menampilkan jumlah data untuk memastikan pembagian yang benar
print(f"Jumlah train dataset: {len(train_data)} ({train_percentage:.2f}%)")
print(f"Jumlah test dataset: {len(test_data)} ({test_percentage:.2f}%)")
print(f"Jumlah validation dataset: {len(validation_data)} ({validation_percentage:.2f}%)")

# %% [markdown]
# ## 6. Modeling and Result

# %% [markdown]
# ### 6.1. Content-Based Filtering
# Pertama, saya akan melakukan modelling sistem rekomendasi berdasarkan konten, atau disebut Content-Based Filtering.

# %% [markdown]
# #### 6.1.1. Hitung Similarity antar Film
# - Menerapkan Cosine Similarity untuk menghitung seberapa mirip film satu sama lain berdasarkan genrenya:

# %%
cos_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)
cos_sim

# %% [markdown]
# #### 6.1.2. Membuat Mapping Hasil Cosine Similarity dan Judul Filmnya
# - Agar bisa mencari film berdasarkan judul:

# %%
cos_df = pd.DataFrame(cos_sim, columns=data['title'], index=data['title'])
cos_df

# %% [markdown]
# #### 6.1.3. Fungsi Rekomendasi
# - Buat fungsi untuk mengambil rekomendasi berdasarkan judul film:

# %%
def film_recommendations(nama_film, similarity_data, items, top_n=10, show_result=True, genre_filter=None):

    # Validasi input
    if similarity_data is None or not isinstance(similarity_data, (np.ndarray, pd.DataFrame)):
        raise ValueError("Parameter 'similarity_data' harus berupa array numpy atau DataFrame.")
    if items is None or 'title' not in items.columns:
        raise ValueError("Parameter 'items' harus berupa DataFrame yang mengandung kolom 'title'.")

    # Salin data untuk menghindari modifikasi langsung
    items = items.copy()
    items['title_lower'] = items['title'].str.lower()
    nama_film = nama_film.lower()

    # Cek apakah film ada dalam dataset
    matches = items[items['title_lower'] == nama_film]
    if matches.empty:
        return f"❌ Film '{nama_film}' tidak ditemukan dalam data."

    # Jika ada lebih dari satu film dengan judul yang sama
    if len(matches) > 1:
        print("⚠️ Beberapa film dengan judul serupa ditemukan. Menggunakan yang pertama ditemukan.")

    # Ambil indeks film
    film_index = matches.index[0]

    # Ambil skor similarity
    similar_scores = similarity_data[film_index]

    # Urutkan dan ambil top_n
    similar_indices = np.argsort(similar_scores)[-top_n-1:-1][::-1]

    # Ambil data film yang mirip
    similar_movies_df = items.iloc[similar_indices][['title', 'genres']].copy()
    similar_movies_df['similarity_score'] = [similar_scores[i] for i in similar_indices]

    # Filter berdasarkan genre jika diperlukan
    if genre_filter:
        similar_movies_df = similar_movies_df[similar_movies_df['genres'].str.contains(genre_filter, case=False)]

    # Reset index agar rapi
    similar_movies_df.reset_index(drop=True, inplace=True)

    # Tampilkan hasil jika show_result True
    if show_result:
        print(f"\n🎬 Rekomendasi film mirip dengan '{matches.iloc[0]['title']}':\n")
        for i, row in similar_movies_df.iterrows():
            print(f"{i+1}. {row['title']} | Genre: {row['genres']} | Similarity: {row['similarity_score']:.2f}")

    return similar_movies_df


# %% [markdown]
# #### 6.1.4. Pengujian Sistem Rekomendasi
# - Pada tahap ini, akan dilakukanlah pengujian dari sistem yang telah dibuat

# %%
# cek data film yang ingin direkomendasikan
data[data['title'] == 'The Matrix']

# %%
title_based_recom = film_recommendations("The Matrix", similarity_data=cos_sim, items=data[['title', 'genres']], top_n=10)
title_based_recom

# %% [markdown]
# - Dapat dilihat dengan menampilkan 10 rekomendasi dengan judul film **The Matrix**, genrenya sama semua, dan memiliki `similarity_score` 1.0

# %% [markdown]
# ### 6.2. Collaborative Filtering
# Setelah membangun Content-Based Filtering, sekarang saya akan melakukan modelling sistem rekomendasi berdasarkan pola interaksi user, atau disebut Collaborative Filtering.

# %% [markdown]
# #### 6.2.1. Membangun Arsitektur Model Deep Learning

# %%
# Ukuran vektor embedding
embedding_size = 50

# Input untuk user dan movie
user_input = Input(shape=(1,), name='user_input')
movie_input = Input(shape=(1,), name='movie_input')

# Embedding user
user_embedding = Embedding(
    input_dim=preparation_cf['user'].nunique(),
    output_dim=embedding_size,
    embeddings_initializer='he_normal',
    embeddings_regularizer=l2(1e-6)
)(user_input)

# Embedding movie
movie_embedding = Embedding(
    input_dim=preparation_cf['movie'].nunique(),
    output_dim=embedding_size,
    embeddings_initializer='he_normal',
    embeddings_regularizer=l2(1e-6)
)(movie_input)

# Bias user dan movie
user_bias = Embedding(
    input_dim=preparation_cf['user'].nunique(),
    output_dim=1
)(user_input)

movie_bias = Embedding(
    input_dim=preparation_cf['movie'].nunique(),
    output_dim=1
)(movie_input)

# Dot product user dan movie
dot_product = Dot(axes=2)([user_embedding, movie_embedding])

# Tambahkan bias
add_bias = Add()([dot_product, user_bias, movie_bias])

# Flatten
x = Flatten()(add_bias)

# Output sigmoid (asumsi rating dinormalisasi 0-1)
output = Dense(1, activation='sigmoid')(x)

# Bangun model
model = Model(inputs=[user_input, movie_input], outputs=output)

# %%
# Kompilasi model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mean_absolute_error', 'root_mean_squared_error']
)

# Tampilkan ringkasan model
model.summary()

# %% [markdown]
# #### 6.2.2. Melatih Model

# %%
# Training
history = model.fit(
    x=[train_data['user'], train_data['movie']],
    y=train_data['rating_normalized'],
    epochs=15,
    batch_size=32,
    validation_data=([validation_data['user'], validation_data['movie']], validation_data['rating_normalized']),
    verbose=1
)

# %% [markdown]
# #### 6.2.3. Pengujian Sistem Rekomendasi

# %%
def recommend_movies_for_user(user_id, model, data_cf, movie_df, top_n=10):
    """
    1. Top 5 film yang pernah dirating tertinggi oleh user
    2. Top-N rekomendasi film berdasarkan prediksi model

    Parameters:
    - user_id: int, ID pengguna yang ingin direkomendasikan film
    - model: model deep learning yang sudah dilatih
    - data_cf: DataFrame, data training yang sudah diproses (preparation_cf)
    - movie_df: DataFrame, data film (harus mengandung 'movieId', 'title', dan 'genres')
    - top_n: int, jumlah rekomendasi film yang ingin ditampilkan

    Returns:
    - top_rated_by_user: DataFrame
    - recommendations: DataFrame
    """

     # ----- Top 5 Film yang Pernah Dirating -----
    user_rated_df = data_cf[data_cf['user'] == user_id].copy()
    user_rated_df = user_rated_df.sort_values(by='rating', ascending=False).head(5)

    top_rated_by_user = user_rated_df.merge(movie_df, left_on='movie', right_on='movieId')
    top_rated_by_user = top_rated_by_user[['title', 'genres', 'rating']]

    # ----- Rekomendasi Film -----
    all_movie_ids = data_cf['movie'].unique()
    rated_movie_ids = user_rated_df['movie'].unique()
    unrated_movie_ids = np.setdiff1d(all_movie_ids, rated_movie_ids)

    user_array = np.full(len(unrated_movie_ids), user_id)
    movie_array = np.array(unrated_movie_ids)

    predicted_ratings = model.predict([user_array, movie_array], verbose=0)

    recommendations = pd.DataFrame({
        'movie': movie_array,
        'predicted_rating': predicted_ratings.flatten()
    })

    recommendations = recommendations.merge(movie_df[['movieId', 'title', 'genres']],
                                            left_on='movie', right_on='movieId')

    recommendations = recommendations.sort_values(by='predicted_rating', ascending=False)
    recommendations = recommendations[['title', 'genres', 'predicted_rating']].head(top_n)

    return top_rated_by_user, recommendations

# %%
user_id = 1  # Define the user_id variable

top5_rated, top10_recommendations = recommend_movies_for_user(
    user_id=user_id,
    model=model,
    data_cf=preparation_cf,
    movie_df=df_movies,
    top_n=10
)

# %%
print(f"Top 5 Film dengan Rating Tertinggi oleh User : {user_id}")
top5_rated

# %%
print("\nTop 10 Film Rekomendasi untuk User : {}".format(user_id))
top10_recommendations

# %% [markdown]
# ## 7. Evaluation

# %% [markdown]
# ### 7.1. Content-Based Filtering

# %% [markdown]
# - Evaluasi kinerja sistem rekomendasi dilakukan untuk mengukur seberapa baik sistem dalam memberikan rekomendasi yang relevan dan sesuai dengan kebutuhan pengguna. Metrik evaluasi yang digunakan dalam menilai kualitas sistem rekomendasi content-based adalah precision, dan recall.
# 
# - Precision adalah metrik evaluasi yang mengukur seberapa akurat sistem dalam merekomendasikan film. Semakin tinggi precision, semakin sedikit film yang tidak relevan yang direkomendasikan. Precision tinggi berarti pengguna tidak dibanjiri oleh film yang tidak relevan, serta rekomendasi terasa lebih personal dan tepat sasaran.
# 
# - Recall menunjukkan seberapa baik model dalam mendeteksi seluruh data positif yang sebenarnya. Recall (juga dikenal sebagai True Positive Rate) mengukur proporsi item yang relevan dan berhasil direkomendasikan dibandingkan dengan semua item yang seharusnya direkomendasikan. Recall tinggi berarti sistem tidak melewatkan banyak film relevan, yang penting agar pengguna tidak kehilangan konten yang mereka sukai.

# %%
# Daftar genre yang dianggap penting sesuai preferensi
target_genres = {"Action", "Science", "Fiction"}

# Fungsi untuk mengecek apakah semua genre target ada dalam genre film
def cocok_dengan_preferensi(genre_str):
    film_genres = set(genre_str.split())
    return target_genres.issubset(film_genres)

# Tandai film yang sesuai preferensi genre
title_based_recom['sesuai_genre'] = title_based_recom['genres'].apply(cocok_dengan_preferensi)

# Hitung jumlah film yang benar-benar sesuai (positif relevan)
jumlah_sesuai = title_based_recom['sesuai_genre'].sum()

# Hitung total film yang diberikan sebagai rekomendasi
jumlah_direkomendasikan = title_based_recom.shape[0]

# Precision: proporsi rekomendasi yang sesuai dari semua yang diberikan
presisi = jumlah_sesuai / jumlah_direkomendasikan * 100

# Hitung total film relevan dalam dataset
total_relevan = title_based_recom[title_based_recom['sesuai_genre']].shape[0]

# Recall: proporsi film relevan yang berhasil direkomendasikan
recall = jumlah_sesuai / total_relevan * 100 if total_relevan > 0 else 0

# Tampilkan hasil
print(f"Precision: {presisi:.2f}%")
print(f"Recall: {recall:.2f}%")
title_based_recom[['title', 'genres', 'sesuai_genre']]

# %% [markdown]
# - Insight:
#   - Precision: 100% (semua rekomendasi relevan dengan preferensi genre)  
#   - Recall: 100% (semua film relevan berhasil direkomendasikan)

# %% [markdown]
# ### 7.2. Collaborative Filtering
# - Evaluasi kinerja sistem rekomendasi dilakukan untuk mengukur seberapa baik sistem dalam memberikan rekomendasi yang relevan dan sesuai dengan kebutuhan pengguna. Metrik evaluasi yang digunakan dalam sistem rekomendasi Collaborative Filtering adalah Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE).  
# 
# - Untuk perhitungannya semakin kecil nilai MAE atau RMSE maka semakin baik kemampuan sistem dalam memprediksi rating pengguna.
# 
# - MAE mengukur rata-rata selisih absolut antara rating yang diprediksi dan rating sebenarnya. MAE menunjukkan seberapa “salah” prediksi model secara rata-rata.
# 
# - RMSE menghitung akar dari rata-rata kuadrat dari error prediksi. RMSE menghukum kesalahan besar lebih keras dibanding MAE (karena error dikuadratkan).
# 

# %%
# evaluasi model dengan test dataset menggunakan MAE dan RMSE
loss, mae, rmse = model.evaluate([test_data['user'], test_data['movie']], test_data['rating_normalized'])
print(f'Evaluasi MAE pada test dataset: {mae}')
print(f'Evaluasi RMSE pada test dataset: {rmse}')

# %%
df_result_mae = pd.DataFrame({'Train Mean Absolute Error':history.history['mean_absolute_error'],
                          'Val Mean Mean Absolute Error':history.history['val_mean_absolute_error']}, index=range(1,16))

sns.lineplot(data=df_result_mae)
plt.title('Metrik Evaluasi MAE Terhadap Data Training dan Validation')
plt.xlabel('epochs')
plt.ylabel('score')
plt.show()

# %%
df_result_rmse = pd.DataFrame({'Train Root Mean Square Error':history.history['root_mean_squared_error'],
                          'Val Root Mean Square Error':history.history['val_root_mean_squared_error']}, index=range(1,16))

sns.lineplot(data=df_result_rmse)

plt.title('Metrik Evaluasi RMSE Terhadap Data Training dan Validation')
plt.xlabel('epochs')
plt.ylabel('score')
plt.show()

# %% [markdown]
# Insight:
# 1. Mean Absolute Error (MAE)
# 
# * Terlihat bahwa nilai **train MAE** menurun konsisten seiring bertambahnya epoch, menunjukkan bahwa model belajar dengan baik terhadap data latih.
# * **Validation MAE** juga menurun dengan kecepatan yang lambat tapi stabil, lalu stagnan di sekitar epoch ke-8 hingga ke-15. Ini menandakan model mulai mengalami **diminishing returns**, namun **belum overfitting** secara signifikan.
# 
# 
# 2. Root Mean Square Error (RMSE)
# 
# * **Train RMSE** juga menurun tajam, dan **val RMSE** stabil turun lalu mendatar. Polanya mirip dengan MAE, artinya model cukup stabil dan tidak mengalami overfitting berat.
# * RMSE lebih sensitif terhadap **outlier** dibanding MAE, jadi grafik ini membantu mendeteksi lonjakan kesalahan, tapi dalam kasusmu tetap terlihat smooth dan terkendali.


