import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image
from keras.models import load_model

class App:

    def __init__(self):
        self.canvaswidth = 280
        self.canvasheight = 280
        self.pixelsize = 7
        self.imageshape = (1, 784)
        self.draw = {}
        self.brushwidth = 5
        self.temp = 0
        self.model = load_model('model.h5')
        self.root = None
        self.canvas = None

    def CreateWindow(self):
        self.root = tk.Tk()
        self.root.title("MNIST Digit Recognition")
        self.frame = tk.Frame(self.root)
        self.canvas = tk.Canvas(self.frame, width = self.canvaswidth, height = self.canvasheight, bg = "black", cursor="cross")
        self.label = ttk.Label(self.frame, text=" Predict Digit", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self.frame, text = " Recognise ", command = self.Predict) 
        self.button_clear = tk.Button(self.frame, text = " Clear ", command = self.ClearCanvas)
        self.frame.pack(pady = 20, padx = 60, fill = "both")
        self.canvas.grid(row=0, column=0, pady=2)
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

    def CreateCanvasGrid(self):
        item_id = 0
        for x in range(0, self.canvaswidth, self.pixelsize):
            for y in range(0, self.canvasheight, self.pixelsize):
                item_id = self.canvas.create_rectangle(
                    x, y, x + self.pixelsize, y + self.pixelsize,
                    fill = "black", outline = "black"
                )
                self.canvas.tag_bind(item_id)
                item_id = item_id - 1600 * self.temp
                self.draw.update({(int((item_id - 1) // (self.canvaswidth / self.pixelsize)), 
                                        int((item_id - 1)  % (self.canvaswidth / self.pixelsize))) : 0})
    
    def GetItemIds(self, event= None, *, dx, dy):
        mx, my = event.x, event.y
        return self.canvas.find_overlapping(mx, my, mx + dx, my + dy)

    def ColorPixel(self, event = None):
        Pixels = self.GetItemIds(event, dx = self.brushwidth, dy = self.brushwidth)
        for px in Pixels:
            self.canvas.itemconfig(px, {"fill" : "white"})
            px = px - 1600 * self.temp
            self.draw.update({(int((px - 1) // (self.canvaswidth / self.pixelsize)),
                                    int((px - 1)  % (self.canvaswidth / self.pixelsize))) : 255})
                      
    def ClearCanvas(self):
        self.canvas.delete("all")
        self.temp += 1
        self.CreateCanvasGrid()
        self.canvas.bind("<B1-Motion>", self.ColorPixel)

    def ExtractImage(self):
        image = np.zeros((int(self.canvaswidth / self.pixelsize), 
                          int(self.canvasheight / self.pixelsize)))
        
        for (px, py), pval in self.draw.items():
            image[(px, py)] = pval

        image = np.array([np.uint8(image).T]) / 255.0
        image = Image.fromarray(image[0]).resize((28, 28))
        image = np.asarray(image).reshape(self.imageshape)

        return image

    def Predict(self):
        drawing = self.ExtractImage()
        prediction = self.model.predict(drawing, verbose = 0)
        prediction_confidence = max(list(zip(range(0, len(prediction.flatten())), prediction.flatten())), key = lambda x : x[1])
        self.label.configure(text= f" {np.argmax(prediction)},{prediction_confidence[1] * 100.0 : .2f}%")
    
    def Display(self):
        self.CreateWindow()
        self.CreateCanvasGrid()
        self.canvas.bind("<B1-Motion>", self.ColorPixel)
        self.root.mainloop()

if __name__ == "__main__":
    app = App()
    app.Display()