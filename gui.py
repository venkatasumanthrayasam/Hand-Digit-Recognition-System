import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("digit_model.h5")

def predict_digit(img):
    """Preprocess image & predict digit"""
    img = img.resize((28,28))          # Resize to 28x28
    img = ImageOps.grayscale(img)      # Convert to grayscale
    img = np.array(img)
    img = img.reshape(1,28,28,1) / 255.0
    result = model.predict(img)
    return np.argmax(result), max(result[0])

def draw(event):
    """Draw on canvas"""
    x, y = event.x, event.y
    canvas.create_oval((x-10, y-10, x+10, y+10), fill="black")

def clear_canvas():
    """Clear the canvas"""
    canvas.delete("all")
    label.config(text="Prediction: None")

def classify():
    """Grab canvas image & classify"""
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Capture canvas
    img = ImageGrab.grab().crop((x, y, x1, y1))
    digit, acc = predict_digit(img)
    label.config(text=f"Prediction: {digit} (Accuracy: {int(acc*100)}%)")

# GUI setup
root = tk.Tk()
root.title("Handwritten Digit Recognition")

canvas = tk.Canvas(root, width=200, height=200, bg="white")
canvas.grid(row=0, column=0, pady=2, sticky="W")
canvas.bind("<B1-Motion>", draw)

btn_classify = tk.Button(root, text="Predict", command=classify)
btn_classify.grid(row=1, column=0, pady=2)

btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.grid(row=2, column=0, pady=2)

label = tk.Label(root, text="Prediction: None", font=("Helvetica", 14))
label.grid(row=3, column=0, pady=2)

root.mainloop()
