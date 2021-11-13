import os
import pygubu
import tensorflow as tf
from crack_detection.model import Model
from tkinter import filedialog
from PIL import ImageTk, Image
from datetime import datetime


class CrackDetectionApp:

    def __init__(self):
        # 1: Create a builder
        self.builder = builder = pygubu.Builder()

        # 2: Load an ui file
        builder.add_from_file('crack_detection.ui')

        self.mainwindow = builder.get_object('main_window')
        self.canvas_predict = builder.get_object('canvas_predict')
        self.inputImg = None
        self.canvas_result = builder.get_object('canvas_result')
        self.predictedImg = None
        self.imageFilename = None
        self.status = builder.get_object('label_status')
        self.scale = builder.get_object('scale')
        self.modelFileName = None
        self.model = None
        self.modelStatus = builder.get_object('label_model')
        self.safeFilename = None
        self.saveStatus = builder.get_object('label_save_image')
        self.mainwindow.iconbitmap('icon.ico')

        # Connect method callbacks
        builder.connect_callbacks(self)

    def run(self):
        self.mainwindow.mainloop()

    # define the method callbacks
    def on_save_image_clicked(self):
        if self.predictedImg is not None:
            if not os.path.exists('output/'):
                os.makedirs('output/')
            now = datetime.now().strftime("%d%m%Y-%H%M%S")
            image = ImageTk.getimage(self.predictedImg)
            image.save(f'output/{now}.png', 'PNG')
            self.saveStatus.config(text=f'Saved to output/{now}.png')
        else:
            self.saveStatus.config(text='Nothing to be saved!')

    def on_choose_image_clicked(self):
        self.imageFilename = filedialog.askopenfilename(initialdir="/",
                                                        title="Select an Image",
                                                        filetypes=[('Image Files',
                                                                    ('.png', '.jpg')),
                                                                   ],
                                                        )

        self.inputImg = ImageTk.PhotoImage(Image.open(self.imageFilename).resize((800, 600)))
        self.canvas_predict.create_image(0, 0, image=self.inputImg, anchor='nw')
        self.canvas_result.delete('all')
        self.predictedImg = None
        self.status.config(text='Ready to detect!')

    def on_choose_model_clicked(self):
        self.modelFileName = filedialog.askopenfilename(initialdir="/",
                                                        title="Select a Model",
                                                        filetypes=[('Models',
                                                                    '.h5'),
                                                                   ],
                                                        )
        self.modelStatus.config(text='Loading...')
        self.modelStatus.update()
        with tf.device('/device:GPU:0'):
            self.model = Model()
            self.model.load_model(self.modelFileName)

        self.modelStatus.config(text=os.path.basename(self.modelFileName))

    def on_detect_clicked(self):
        if self.model is not None:
            self.status.config(text='Detecting...')
            self.status.update()
            with tf.device('/device:GPU:0'):
                predictions, crack_status = self.model.predict_on_crops(self.imageFilename,
                                                                        ['crack', 'non crack'],
                                                                        width=int(self.scale.get()),
                                                                        height=int(self.scale.get())
                                                                        )
                self.predictedImg = ImageTk.PhotoImage(Image.fromarray(predictions).resize((800, 600)))
                self.canvas_result.create_image(0, 0, image=self.predictedImg, anchor='nw')
            if crack_status:
                self.status.config(text='Detected!')
            else:
                self.status.config(text='Undetected')
        else:
            self.status.config(text='Please choose a model!')

    def accept_whole_number_only(self, e=None):
        value = self.scale.get()
        if int(value) != value:
            self.scale.set(round(value))

    def reset_scale(self):
        self.scale.set(256)


if __name__ == '__main__':
    app = CrackDetectionApp()
    app.run()
