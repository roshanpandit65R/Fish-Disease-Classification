from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Ensure model file exists before loading
model_path = "models/fish_disease_model.h5"

if os.path.exists(model_path):
    print("Model file found. Loading model...")
    model = tf.keras.models.load_model(model_path)
else:
    print("Error: Model file not found. Please train the model first.")
    exit()

# Define class names (should match train_data.class_indices)
class_names = ['Bacterial diseases - Aeromoniasis', 'Bacterial gill disease', 'Bacterial Red disease', 
               'Fungal diseases Saprolegniasis', 'Healthy Fish', 'Parasitic diseases', 'Viral diseases White tail disease']

# Function to process and predict image
def predict_disease(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return class_names[class_index]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        
        file = request.files["file"]
        if file.filename == "":
            return "No file selected"

        # Ensure static directory exists
        os.makedirs("static", exist_ok=True)

        # Save uploaded file
        file_path = os.path.join("static", file.filename)
        file.save(file_path)

        # Predict disease
        result = predict_disease(file_path)
        return render_template("result.html", result=result, image=file.filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
