import io
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tkinter as tk

def load_model():
    # Charger le modèle
    model = tf.keras.models.load_model('mnist_model.h5')
    return model

def predict_digit(model, image):
    # Convertir l'image en niveaux de gris
    image_gray = ImageOps.grayscale(image)
    print("Image en niveaux de gris obtenue")

    # Redimensionner l'image à la taille requise (28x28)
    image_resized = image_gray.resize((28, 28))
    print("Image redimensionnée à 28x28")

    # Convertir l'image en un tableau numpy et normaliser les valeurs des pixels
    image_array = np.array(image_resized) / 255.0
    print(f"Image convertie en tableau numpy et normalisée : {image_array.shape}")

    # Ajouter une dimension pour correspondre à la forme attendue par le modèle
    image_array = image_array[np.newaxis, ..., np.newaxis]
    print(f"Image avec dimension ajoutée : {image_array.shape}")

    # Faire la prédiction
    prediction = model.predict(image_array)
    print(f"Prédiction brute : {prediction}")
    digit = np.argmax(prediction)
    print(f"Chiffre prédit : {digit}")
    return digit

def capture_image(canvas: tk.Canvas):
    # Capturer l'image dessinée sur le canvas
    ps = canvas.postscript(colormode='color')
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    return img