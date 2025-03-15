from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("saved_model.h5")  # Make sure this file exists

# Define class labels (Modify based on your model)
CLASS_LABELS = ["Normal", "Cataract", "Diabetic Retinopathy", "Glaucoma"]

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # Ensure correct input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img /= 255.0  # Normalize pixel values
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file selected!")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected!")

        if file:
            file_path = os.path.join("static/uploads", file.filename)
            file.save(file_path)

            try:
                # Preprocess the image
                img = preprocess_image(file_path)

                # Get prediction
                preds = model.predict(img)
                class_idx = np.argmax(preds)  # Get the index of highest probability
                prediction = CLASS_LABELS[class_idx]

            except Exception as e:
                prediction = f"Error in prediction. Try again! ({str(e)})"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
