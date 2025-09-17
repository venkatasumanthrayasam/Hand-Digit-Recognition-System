import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, Image, ImageOps
import numpy as np
import tensorflow as tf

# Load the trained model
try:
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
except OSError:
    messagebox.showerror("Model not found", "Please run train_model.py first to train and save the model.")
    exit()

class DigitRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.geometry("320x420")
        self.resizable(False, False)

        self.canvas = tk.Canvas(self, width=280, height=280, bg="black", cursor="dot")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<ButtonRelease-1>", self.reset_pos)

        self.label = tk.Label(self, text="Draw a digit (0-9)", font=("Helvetica", 16))
        self.label.pack(pady=5)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=5)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10)

        self.lastx, self.lasty = None, None

    def draw_digit(self, event):
        x, y = event.x, event.y
        if self.lastx and self.lasty:
            self.canvas.create_line(self.lastx, self.lasty, x, y,
                                    width=18, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE)
        self.lastx, self.lasty = x, y

    def reset_pos(self, event):
        self.lastx, self.lasty = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.lastx, self.lasty = None, None
        self.label.config(text="Draw a digit (0-9)")

    def predict_digit(self):
        # Capture canvas content
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        image = ImageGrab.grab(bbox=(x, y, x1, y1))

        # Preprocess image
        image = image.resize((28, 28)).convert('L')
        img_array = np.array(image).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        self.label.config(text=f"Prediction: {predicted_digit} ({confidence:.2f}%)")


if __name__ == "__main__":
    app = DigitRecognizer()
    app.mainloop()