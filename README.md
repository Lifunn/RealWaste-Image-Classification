# Klasifikasi Gambar Sampah (RealWaste Image Classification) dengan CNN

Proyek ini adalah implementasi dari *Convolutional Neural Network* (CNN) yang dibangun untuk mengklasifikasikan gambar berbagai jenis sampah. Model ini dilatih untuk mengenali dan membedakan antara [**Isi jumlah kelas, misal: 6 kategori sampah**] yang berbeda, menggunakan dataset "RealWaste".

Repositori ini mencakup *notebook* Jupyter (`CNN model.ipynb`) yang berisi seluruh alur kerja, mulai dari pemrosesan data, definisi arsitektur model, pelatihan, hingga evaluasi performa.

## Live Demo

Aplikasi demo interaktif untuk proyek ini telah di-deploy menggunakan Hugging Face Spaces.

**Anda dapat mengakses dan mencoba model ini secara langsung di:**
[**https://huggingface.co/spaces/Lifunn/Demo-ML**](https://huggingface.co/spaces/Lifunn/Demo-ML)

---

## Tumpukan Teknologi (Tech Stack)

* **Bahasa:** Python
* **Framework Deep Learning:** [Sebutkan framework: misal, TensorFlow/Keras atau PyTorch]
* **Library:** NumPy, Matplotlib, Scikit-learn (untuk evaluasi), OpenCV (jika digunakan)

---

## Dataset

Model ini dilatih menggunakan dataset **RealWaste**.

* **Deskripsi:** [Jelaskan singkat datasetnya, misal: "Dataset ini terdiri dari gambar-gambar sampah yang telah dikategorikan ke dalam kelas organik dan anorganik..."].
* **Jumlah Kelas:** [Isi jumlah kelas, misal: 6]
* **Nama Kelas:** [Sebutkan nama-nama kelasnya, misal: 'kaca', 'kertas', 'plastik', 'logam', 'organik', 'lain-lain']
* **Ukuran Gambar:** [Isi resolusi gambar yang digunakan, misal: 224x224x3]

---

## Arsitektur Model

Arsitektur CNN yang digunakan terdiri dari lapisan-lapisan berikut:

1.  **Input Layer:** [Sebutkan input shape, misal: (224, 224, 3)]
2.  **Convolutional Layer 1:** [Contoh: 32 filter, kernel (3x3), aktivasi 'relu']
3.  **Pooling Layer 1:** [Contoh: MaxPooling (2x2)]
4.  **Convolutional Layer 2:** [Contoh: 64 filter, kernel (3x3), aktivasi 'relu']
5.  **Pooling Layer 2:** [Contoh: MaxPooling (2x2)]
6.  **Flatten Layer**
7.  **Dense (Fully Connected) Layer:** [Contoh: 128 unit, aktivasi 'relu']
8.  **Output Layer (Dense):** [Contoh: 6 unit (sesuai jumlah kelas), aktivasi 'softmax']

*(Sesuaikan deskripsi di atas agar cocok dengan arsitektur model di notebook Anda)*

---

## Hasil dan Performa

Model dilatih selama [**Jumlah Epoch, misal: 20**] epoch menggunakan optimizer [**Nama Optimizer, misal: 'Adam'**] dan *loss function* [**Nama Loss, misal: 'categorical_crossentropy'**].

Performa model pada set data pengujian (test set) adalah sebagai berikut:

* **Test Accuracy (Akurasi):** **79.42%**
* **Test Loss (Loss):** **0.570**

### Kurva Pelatihan
<img width="706" height="565" alt="image" src="https://github.com/user-attachments/assets/be87c701-117d-4f52-9128-60af6bc5a08c" />

<img width="706" height="568" alt="image" src="https://github.com/user-attachments/assets/cb7ef81e-6799-49dc-8353-990a764b5346" />


---
