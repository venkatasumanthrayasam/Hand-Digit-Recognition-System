import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, Image, ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self.geometry("900x800")   # 🔥 Bigger window
        self.resizable(False, False)

        # Canvas for drawing (bigger now)
        self.canvas = tk.Canvas(self, width=400, height=400, bg="black", cursor="dot")
        self.canvas.pack(side=tk.LEFT, padx=20, pady=20)
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<ButtonRelease-1>", self.reset_pos)

        # Right panel
        right_frame = tk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.label = tk.Label(right_frame, text="Draw a digit (0-9)", font=("Helvetica", 20))
        self.label.pack(pady=15)

        self.button_frame = tk.Frame(right_frame)
        self.button_frame.pack(pady=10)

        self.predict_button = tk.Button(self.button_frame, text="Predict", font=("Helvetica", 14),
                                        width=12, command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.clear_button = tk.Button(self.button_frame, text="Clear", font=("Helvetica", 14),
                                      width=12, command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10)

        # Graph placeholder (bigger)
        self.fig, self.ax = plt.subplots(figsize=(6, 4))  # 🔥 larger plot
        self.ax.set_title("Prediction Probabilities", fontsize=14)
        self.ax.set_xlabel("Digits", fontsize=12)
        self.ax.set_ylabel("Probability", fontsize=12)
        self.bar_container = self.ax.bar(range(10), [0]*10, tick_label=list(range(10)))
        self.fig.tight_layout()

        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_graph.get_tk_widget().pack(pady=20)

        self.lastx, self.lasty = None, None

    def draw_digit(self, event):
        x, y = event.x, event.y
        if self.lastx and self.lasty:
            self.canvas.create_line(self.lastx, self.lasty, x, y,
                                    width=22, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE)  # thicker pen
        self.lastx, self.lasty = x, y

    def reset_pos(self, event):
        self.lastx, self.lasty = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.lastx, self.lasty = None, None
        self.label.config(text="Draw a digit (0-9)")
        # Reset graph
        for bar in self.bar_container:
            bar.set_height(0)
        self.canvas_graph.draw()

    def predict_digit(self):
        # Capture canvas content
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        image = ImageGrab.grab(bbox=(x, y, x1, y1))

        # Preprocess image (unchanged)
        image = image.resize((28, 28)).convert('L')
        img_array = np.array(image).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        self.label.config(text=f"Prediction: {predicted_digit} ({confidence:.2f}%)")

        # Update graph
        for bar, p in zip(self.bar_container, prediction[0]):
            bar.set_height(p)
        self.ax.set_ylim(0, 1)  # probabilities between 0–1
        self.canvas_graph.draw()


if __name__ == "__main__":
    app = DigitRecognizer()
    app.mainloop()
