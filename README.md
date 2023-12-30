# Laporan Proyek Machine Learning

### Nama : Nurlisa Widyaningsih

### Nim : 211351108

### Kelas : IF Pagi B

## Domain Proyek
Web app dengan fungsi melakukan prediksi apakah penghasilan anda seharusnya lebih dari 50,000 USD atau kurang sama dengan 50,000 USD. Bisa digunakan untuk menentukan apakah pekerjaanmu membayarmu dengan bayaran yang sesuai dengan investasi pendidikan dan skill yang telah diperjuangkan olehmu.

## Business Understanding
Semakin banyaknya pekerjaan yang terbuka, maka semakin banyak pula perusahaan/pekerjaan yang tidak membayar karyawannya dengan jumlah yang sesuai dengan skill karyawannya. 

### Problem Statement
Karyawan tidak mendapatkan bayaran yang sesuai dari eemployernya dan karyawan tidak menyadari bahwa dia itu tidak dibayar dengan biaya yang sesuai. 
### Goals
Menyadarkan karyawan mengenai pendapatan yang seharusnya dia dapatkan pertahun.
### Solution Statements
-   Dengan membuat web app yang bisa memprediksi mengenai pendapatan pekerja menggunakan algorithma D-Tree (Decision Tree).

## Data Understanding
Dataset ini diambil dari database sensus biro tahun 1994 oleh Ronny Kohavi dan Barry Becker. Dataset yang diambil ini cukup bersih dan banyak, serta bisa digunakan untuk melakukan prediksi apakah pendapatan seseorang itu mencapai lebih dari 50,000 USD pertahunnya. 
[Census Adult Income](https://www.kaggle.com/datasets/uciml/adult-census-income) 

### Variabel-variabel pada Diabetes Prediction adalah sebagai berikut:
- age : Menunjukkan data umur. (int, 17 hingga 90)
- workclass : Menunjukkan kelas pekerjaan. (categorial, private;state-gov;dll) 
- fnlwgt : Menunjukkan jumlah orang yang diyakini berdasarkan sensus entri tersebut mewakili. (int, lebih dari 0)
- education : Menunjukkan jenjang pendidikan. (categorial, Bachelors; Some-college; dll)
- education.num : Menunjukkan jenjang pendidikan dalam bentuk numeric. (int, lebih dari 0)
- martial.status : Menunjukkan status pernikahan. (categorial, married; divorced; dll)
- occupation : Menunjukkan pekerjaan secara general. (categorial, Tech-support; Craft-repair; dll)
- relationship : Menunjukkan apa hubungan individu dengan orang lain. (categorial, Wife; Husband; dll)
- race : Menunjukkan ras individu. (categorial, White; Asian-Pac-Islander; dll)
- sex : Menunjukkan jenis kelamin individu. (categorial, Male; Female)
- capital.gain : Menunjukkan keuntungan seorang individu (int, lebih dari 0)
- capital.loss : Menunjukkan kerugian seorang individu (int, lebih dari 0)
- hours.per.week : Menunjukkan bayaran individu per jam. (int, lebih dari 0)
- native.country : Menunjukkan asal negara individu. (categorial, United-States; Cambodia; dll)
- income : Menunjukkan penghasilan individu. (int, lebih dari 0)

## Data Preparation
### Import Dataset
Untuk langkah pertama dari segala-galanya, kita akan melakukan pengunduhan datasets.
```python
from google.colab import files
files.upload()
```
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
```python
!kaggle datasets download -d uciml/adult-census-income
```
```python
!unzip adult-census-income.zip -d datasets
!ls datasets
```
### Import library
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import pickle
```

### Data Discovery
Ditahap ini langkah pertama yang akan kita lakukan adalah memasukkan data yang tadi telah diextract kedalam sebuah variable.
```python
df = pd.read_csv('datasets/adult.csv')
```
Melihat apakah terdapat nilai null pada datasets.
```python
df.isnull().sum()
```
Next melihat jumlah data unique pada masing-masing kolom
```python
df.nunique()
```
Terdapat 73 nilai unique age, yaa menandakan range umur pada datasets ini adalah 73.
```python
df['education'].unique()
```
Diatas merupakan data unique yang terdapat pada kolom education, ada 16 tingkat edukasi.
```python
df.info()
```
Datasets ini memiliki 32561 baris data. Cukup banyak untuk dianalisis dengan mendetail. Dengan 15 columns.
```python
df.describe()
```
```python
df.duplicated().sum()
```
Terdapat 24 duplicate data, kemungkinan besar akan kita hilangkan pada proses preprocessing nanti.
### Data Cleansing Dan EDA
```python
sns.scatterplot(x = 'hours.per.week', y= 'age', hue = 'income', data = df)
```
![download](https://github.com/NurlisaWidya/income-prediction/assets/148893422/34a6241a-8898-408f-a37a-974a4d6763a7)<br>
Diatas merupakan plot scatter antara hours.per.week dan age. Bisa dilihat disini kebanyakan orang berumur 20an masih belum berhasil memiliki income diatas 50k.
```python
sns.countplot(x = 'income', hue = 'race', data = df)
```
![download](https://github.com/NurlisaWidya/income-prediction/assets/148893422/ac71482e-513a-4c9b-a3a3-6e18f87f7ae5)<br>
plot diatas menunjukkan bahwa orang putih(Whites) memiliki income lebih dari 50k sedangkan ras lain hanya sedikit.
```python
income_0 = df[df["income"] == '<=50K']
income_1 = df[df["income"] == '>50K']

fig0, ax0 = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
education_counts_0 = income_0["education"].value_counts()
ax0.bar(education_counts_0.index, education_counts_0, color="skyblue")
ax0.set_xlabel("Education Level")
ax0.set_ylabel("Count")
ax0.set_title("Income <=50K")
plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
plt.tight_layout()
plt.show()
```
![download](https://github.com/NurlisaWidya/income-prediction/assets/148893422/b745c6f3-74f1-4cbe-9db6-fb3836607bcc)<br>
Menurut bar plot diatas orang dengan pendapatan dibawah 50k pertahunnya biasanya merupakan lulusan High School(SMA/SMK).
```python
# Plot for income 1
fig1, ax1 = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
education_counts_1 = income_1["education"].value_counts()
ax1.bar(education_counts_1.index, education_counts_1, color="lightgreen")
ax1.set_xlabel("Education Level")
ax1.set_ylabel("Count")
ax1.set_title("Income >50K")
plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
plt.tight_layout()
plt.show()
```
![download](https://github.com/NurlisaWidya/income-prediction/assets/148893422/663a81fe-3b21-4998-82e0-cd577735ac07) <br>
Sedangkan orang berpendapatan diatas 50k pertahunnya merupakan lulusan S1, diikuti dengan High School dan diperingkat 4 ada lulusan Masters. Mungkin karena lulusan Masters itu lebih langka dibandingkan High School-grad. Jadi menurut saya wajar saja jika ianya tidak terlalu banyak. Jika dibandingkan dengan plot income dibawah 50k, magister sangatlahh rendah. Menyimpulkan mayoritas lulusan Master memiliki pendapatan diatas 50k pertahunnya.
```python
education_levels = df["education"].unique()  # Replace with your actual column name

# Calculate average capital gain for each education level
avg_capital_gain = df.groupby("education")["capital.gain"].mean()  # Replace with your actual column names

# Create the bar plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.bar(education_levels, avg_capital_gain, color="skyblue")

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Add labels and title
plt.xlabel("Education Level")
plt.ylabel("Average Capital Gain")
plt.title("Average Capital Gain by Education Level")

plt.tight_layout()
plt.show()
```
![download](https://github.com/NurlisaWidya/income-prediction/assets/148893422/e7d34e3a-05a8-4d8f-9340-a66c2562daaf)<br>
Diatas merupakan korelasi antara kolom education level dan avg capital gain yang bisa didapatkan oleh lulusan tertentu. <br>
Nah, untuk tahap pre-processing, Pertama-tama, kita akan menghilangkan data duplicate yang tadi ditemukan.
```python
df=df.drop_duplicates(keep="first")
df.duplicated().sum()
```
Sudah aman ya, tidak ada data duplicate. <br>
Saat melihat datasetsnya terdapat nilai "?" pada beberapa baris data. Kita akan mengubah nilai-nilai "?" itu dengan nilai modus dari masing-masing kolom.
```python
df["workclass"]=df["workclass"].replace("?",np.nan)
df["occupation"]=df["occupation"].replace("?",np.nan)
df["native.country"]=df["native.country"].replace("?",np.nan)

df.isna().sum()
```
Bisa dilihat yaa, terdapat 1836 data yang null pada workclass, 1843 data null pada occupation dan 582 data null pada native.country.
```python
df["workclass"]=df["workclass"].fillna(df["workclass"].mode()[0])
df["occupation"]=df["occupation"].fillna(df["occupation"].mode()[0])
df["native.country"]=df["native.country"].fillna(df["native.country"].mode()[0])
```
Kita memasukkan nilai modusnya hanya pada nilai-nilai yang null.
<br>
Kita akan mengubah nilai preschool hingga grade 12 yang ada pada education dan menggantinya menjadi school, agar tidak terlalu banyak data unique. Kita juga akan menghapus kolom education.num karena ianya memberikan data/informasi yang sama seperti kolom education. Kita lakukan ini agar tidak ada data redudansi.
```python
df.drop(['education.num'], axis = 1, inplace = True)
df['education'].replace(['11th', '9th', '7th-8th', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'],' School', inplace = True)
df['education'].value_counts()
```
```python
df  = df.drop("sex", axis=1)
df  = df.drop("race", axis=1)
df  = df.drop("native.country", axis=1)
df = df.drop("fnlwgt", axis=1)
```
Menghapus kolom sex, race, native.country, dan fnlwgt karena tidak terlalu berpengaruh dengan hasil modelnya nanti.
```python
X = df.drop(['income'], axis=1)
Y = df['income']
```
Memasukkan fitur dan targetnya untuk pemodelan nanti.
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

le = LabelEncoder()
categorical = ['workclass','education', 'marital.status', 'occupation', 'relationship']
for feature in categorical:
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])
```
## Modeling
Melakukan pemodelan dengan criterion entropy.
```python
model_dtree = DecisionTreeClassifier(max_depth=4, criterion="entropy")
model_dtree.fit(X_train, Y_train)
Y_pred = model_dtree.predict(X_test)
```
```python
print(Y_pred[0:5])
print(Y_test[0:5])
```
### Visualisasi hasil algoritma
```python
fitur = X_train.columns
target = df['income'].unique().tolist()
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model_dtree,
                   feature_names=fitur,
                   class_names=target,
                   filled=True)
```
![download](https://github.com/NurlisaWidya/income-prediction/assets/148893422/d5c26a2f-852c-4b92-bfcc-ad1e43c41d07)
## Evaluation
Pada tahap evaluasi ini saya menggunakan accuracy_score dan confusion_matrix saja yaa. Untuk melihat score dan juga melihat True Positive, True Negative, False Positive dan False Negative, confusion_matrix ini cocok untuk algorithma yang bersifat klasifikasi.
```python
print('Akurasi Decision Tree :', accuracy_score(Y_test, Y_pred))
```
Nice, kita berhasil mendapatkan 80% yang mana itu merupakan score yang cukup baik. Mari lihat hasil confusion matrixnya.
```python
y_pred = model_dtree.predict(X_test)
confusion_matrix(Y_test, y_pred)
```
UFT, tidak terlihat terlalu baik ya...hasilnya condong lebih pada true positif sedangkan true negatifnya memiliki nilai yang lebih rendah dibanding false negatifnya.
## Deployment
[Aplikasi Prediksi Income](https://income-prediction-ica.streamlit.app/) <br>
![image](https://github.com/NurlisaWidya/income-prediction/assets/148893422/9a14baae-4325-42f9-ac67-98da25fdc957)
