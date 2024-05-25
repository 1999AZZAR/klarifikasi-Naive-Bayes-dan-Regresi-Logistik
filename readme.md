# Klasifikasi Menggunakan Naive Bayes dan Regresi Logistik

Proyek ini bertujuan untuk melakukan klasifikasi menggunakan dua model pembelajaran mesin: Naive Bayes dan Regresi Logistik. Hasil prediksi dan akurasi dari kedua model akan dibandingkan dan divisualisasikan.

## Struktur Data

Data yang digunakan dalam proyek ini harus dalam format CSV dengan kolom-kolom berikut:

- `Umur`: Usia individu (numerik)
- `Jenis Kelamin`: Jenis kelamin individu (`Laki-Laki` atau `Perempuan`)
- `Pendapatan (Ribu Rupiah)`: Pendapatan individu dalam ribuan Rupiah (numerik)
- `Label`: Label klasifikasi target (numerik atau kategorikal)

Contoh format data dalam CSV:

```csv
Umur,Jenis Kelamin,Pendapatan (Ribu Rupiah),Label
25,Laki-Laki,35,1
30,Perempuan,45,1
22,Laki-Laki,28,0
35,Laki-Laki,50,1
28,Perempuan,38,1
40,Perempuan,60,1
19,Laki-Laki,20,0
33,Laki-Laki,48,1
29,Perempuan,42,1
45,Laki-Laki,70,1
21,Laki-Laki,25,0
27,Perempuan,40,1
38,Perempuan,55,1
20,Laki-Laki,30,0
32,Laki-Laki,46,1
26,Perempuan,36,1
42,Laki-Laki,65,1
23,Laki-Laki,32,0
31,Perempuan,44,1
36,Perempuan,58,1
```

## Cara Menjalankan

1. **Persiapkan Lingkungan**
   Pastikan Anda memiliki Python dan pip terinstal di sistem Anda. Instal pustaka yang diperlukan dengan menjalankan perintah berikut:

    ```bash
    pip install pandas scikit-learn matplotlib seaborn openpyxl
    ```

2. **Simpan Script Python**
   Simpan kode berikut ke dalam sebuah file bernama `main.py`:

    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load data from CSV file
    data = pd.read_csv('data.csv')

    # Convert categorical 'Jenis Kelamin' column to numerical
    data['Jenis Kelamin'] = data['Jenis Kelamin'].map({'Laki-Laki': 0, 'Perempuan': 1})

    # Split the data into features (X) and target (y)
    X = data[['Umur', 'Jenis Kelamin', 'Pendapatan (Ribu Rupiah)']]
    y = data['Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Naive Bayes classifier
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train, y_train)

    # Train a Logistic Regression model
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, y_train)

    # Make predictions using both models
    naive_bayes_predictions = naive_bayes_classifier.predict(X_test)
    logistic_regression_predictions = logistic_regression_model.predict(X_test)

    # Calculate accuracy for both models
    naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
    logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)

    # Print the predictions and accuracies
    print("Naive Bayes Predictions:")
    print(naive_bayes_predictions)

    print("\nLogistic Regression Predictions:")
    print(logistic_regression_predictions)

    print("\nNaive Bayes Accuracy:", naive_bayes_accuracy)
    print("Logistic Regression Accuracy:", logistic_regression_accuracy)

    # Plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Naive Bayes predictions plot
    sns.scatterplot(x=X_test['Umur'], y=X_test['Pendapatan (Ribu Rupiah)'], hue=naive_bayes_predictions, palette='coolwarm', ax=ax[0])
    ax[0].set_title(f'Naive Bayes Predictions\nAccuracy: {naive_bayes_accuracy:.2f}')
    ax[0].set_xlabel('Umur')
    ax[0].set_ylabel('Pendapatan (Ribu Rupiah)')

    # Logistic Regression predictions plot
    sns.scatterplot(x=X_test['Umur'], y=X_test['Pendapatan (Ribu Rupiah)'], hue=logistic_regression_predictions, palette='coolwarm', ax=ax[1])
    ax[1].set_title(f'Logistic Regression Predictions\nAccuracy: {logistic_regression_accuracy:.2f}')
    ax[1].set_xlabel('Umur')
    ax[1].set_ylabel('Pendapatan (Ribu Rupiah)')

    plt.tight_layout()
    plt.show()
    ```

3. **Menjalankan Script**
   Pastikan file data Anda bernama `data.csv` dan berada dalam direktori yang sama dengan `main.py`. Jalankan script dengan perintah:

    ```bash
    python main.py
    ```

## Contoh Penggunaan

Setelah menjalankan script, output terminal akan menunjukkan prediksi dan akurasi untuk masing-masing model:

```text
Naive Bayes Predictions:
[0 0 1 1]

Logistic Regression Predictions:
[1 0 1 1]

Naive Bayes Accuracy: 0.75
Logistic Regression Accuracy: 1.0
```

Selain itu, dua plot akan ditampilkan, masing-masing menunjukkan hasil prediksi Naive Bayes dan Regresi Logistik, beserta akurasinya.

## Penutup

Proyek ini memberikan contoh dasar bagaimana menggunakan Naive Bayes dan Regresi Logistik untuk klasifikasi, serta cara membandingkan kinerja kedua model. Untuk penggunaan lebih lanjut, Anda dapat menyesuaikan parameter model, fitur yang digunakan, atau metode validasi sesuai kebutuhan spesifik Anda.
