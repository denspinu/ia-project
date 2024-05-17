import tkinter as tk
from tkinter import Button
from PIL import Image, ImageOps
import predict
import io

def clear_canvas():
    canvas.delete('all')
    label.config(text="")

def recognize_digit():
    # Capturer l'image dessinée
    image = predict.capture_image(canvas)
    # Faire la prédiction
    digit = predict.predict_digit(model, image)
    label.config(text=f"Digit: {digit}")
    print(f"Digit: {digit}")  # Afficher le chiffre reconnu

def on_mouse_down(event):
    global prev_x, prev_y
    prev_x, prev_y = event.x, event.y

def on_mouse_drag(event):
    global prev_x, prev_y
    canvas.create_line(prev_x, prev_y, event.x, event.y, fill="black", width=5)
    prev_x, prev_y = event.x, event.y

# Créer l'interface utilisateur
root = tk.Tk()
root.title("Handwritten Digit Recognition")

canvas = tk.Canvas(root, width=200, height=200, bg='white', highlightcolor="#000000")
canvas.pack()

label = tk.Label(root, text="")
label.pack()

button_recognize = Button(root, text="Recognize", command=recognize_digit)
button_recognize.pack(side="left")

button_clear = Button(root, text="Clear", command=clear_canvas)
button_clear.pack(side="right")

# Charger le modèle
model = predict.load_model()

canvas.bind("<Button-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_drag)

root.mainloop()