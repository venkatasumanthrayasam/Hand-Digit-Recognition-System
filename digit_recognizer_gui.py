import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, Image, ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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
        self.geometry("600x700")
        self.resizable(False, False)

        # Main frame
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for drawing
        self.canvas = tk.Canvas(main_frame, width=280, height=280, bg="black", cursor="dot")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<ButtonRelease-1>", self.reset_pos)

        # Label for prediction
        self.label = tk.Label(main_frame, text="Draw a digit (0-9)", font=("Helvetica", 16))
        self.label.pack(pady=5)

        # Button frame
        self.button_frame = tk.Frame(main_frame)
        self.button_frame.pack(pady=5)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10)

        # Graph frame
        graph_frame = tk.Frame(main_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create matplotlib figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Initialize bar chart
        self.digits = list(range(10))
        self.probabilities = [0] * 10
        self.bars = self.ax.bar(self.digits, self.probabilities, color='skyblue')
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Digits')
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Prediction Probabilities')
        
        # Add the plot to the tkinter window
        self.canvas_graph = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
        
        # Reset the graph
        for i, bar in enumerate(self.bars):
            bar.set_height(0)
        self.fig.canvas.draw_idle()

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
        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Update the graph
        for i, bar in enumerate(self.bars):
            bar.set_height(prediction[0][i])
        self.fig.canvas.draw_idle()
        
        self.label.config(text=f"Prediction: {predicted_digit} ({confidence:.2f}%)")


if __name__ == "__main__":
    app = DigitRecognizer()
    app.mainloop()