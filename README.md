# 🐟 Fish Disease Classification using AI (Flask + TensorFlow)

This project uses deep learning and image processing to identify common fish diseases from freshwater aquaculture in South Asia. The user can upload an image of a fish, and the system will detect and classify the disease using a trained CNN model.

---

## 📌 Features

- 🧠 CNN-based deep learning model using TensorFlow
- 🖼️ Image upload via a simple Flask web interface
- 🔍 Image preprocessing with OpenCV
- 📊 Trained on 7 fish disease classes (from Kaggle dataset)
- 🧪 Demo support for SVM algorithm (for presentation purpose only)

---

## 📂 Dataset

The dataset contains images of diseased and healthy freshwater fish.  
🔗 **Download it here:**  
[Kaggle - Freshwater Fish Disease Dataset](https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia)

### Disease Classes

- Aeromoniasis
- Bacterial Gill Disease
- Bacterial Red Disease
- Saprolegniasis (Fungal)
- Healthy Fish
- Parasitic Disease
- White Tail Disease (Viral)

---

## 🛠️ Technologies Used

| Tool / Language   | Purpose                         |
|-------------------|----------------------------------|
| Python            | Core programming language        |
| Flask             | Web framework                    |
| TensorFlow / Keras| CNN model creation & prediction  |
| OpenCV            | Image preprocessing              |
| NumPy             | Array operations                 |
| Pillow            | Image saving and format handling |

---

## 📥 Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/Fish_Disease_Classification.git
cd Fish_Disease_Classification
```

### Step 2: Install dependencies

```bash
pip install tensorflow flask opencv-python numpy pillow matplotlib scikit-learn
```

---

## ▶️ Running the Application

1. **Make sure your trained model is saved at:**  
   `models/fish_disease_model.h5`
   ```bash
train_model.py
```

2. **Start the Flask app:**

```bash
python app.py
```

3. **Visit in browser:**  
   `http://127.0.0.1:5000`

Upload a fish image and get a prediction instantly.

---

## 💡 Future Scope

- Improve accuracy using transfer learning
- Add real-time webcam capture
- Integrate SVM and other models for comparison
- Extend support for mobile view or Android app

---

## 🤝 Acknowledgements

- Dataset by Subir Biswas on Kaggle
- TensorFlow and Keras community
- Flask Documentation

---

## 📜 License

This project is for academic and demo use only. Feel free to modify and improve.

