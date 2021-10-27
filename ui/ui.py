import pygubu
import json
import tensorflow as tf
from crack_detection.model import Model
from tkinter import filedialog
from PIL import ImageTk, Image


class CrackDetectionApp:

    def __init__(self):
        # 1: Create a builder
        self.builder = builder = pygubu.Builder()

        # 2: Load an ui file
        builder.add_from_file('crack_detection.ui')

        # 3: Create the mainwindow
        self.mainwindow = builder.get_object('main_window')

        self.canvas_predict = builder.get_object('canvas_predict')
        self.inputImg = None
        self.canvas_result = builder.get_object('canvas_result')
        self.predictedImg = None
        self.filename = None
        self.status = builder.get_object('label_status')
        self.scale = builder.get_object('scale')

        # Connect method callbacks
        builder.connect_callbacks(self)

    def run(self):
        self.mainwindow.mainloop()

    # define the method callbacks
    def on_choose_file_clicked(self):
        self.filename = filedialog.askopenfilename(initialdir="/",
                                                   title="Select an Image",
                                                   filetypes=[('Image Files',
                                                               ('.png', '.jpg')),
                                                              ],
                                                   )

        self.inputImg = ImageTk.PhotoImage(Image.open(self.filename).resize((800, 600)))
        self.canvas_predict.create_image(0, 0, image=self.inputImg, anchor='nw')
        self.canvas_result.delete('all')
        self.status.config(text='Ready to detect!')

    def on_detect_clicked(self):
        self.status.config(text='Detecting...')
        self.status.update()
        with tf.device('/device:GPU:0'):
            configs = json.load(open('../config.json', 'r'))
            model = Model()
            model.load_model(f'../{configs["model"]["load_dir"]}')
            predictions, crack_status = model.predict_on_crops(self.filename,
                                                               configs['data']['classes'],
                                                               width=int(self.scale.get()),
                                                               height=int(self.scale.get())
                                                               )
            self.predictedImg = ImageTk.PhotoImage(Image.fromarray(predictions).resize((800, 600)))
            self.canvas_result.create_image(0, 0, image=self.predictedImg, anchor='nw')
        if crack_status:
            self.status.config(text='Detected!')
        else:
            self.status.config(text='Undetected')

    def accept_whole_number_only(self, e=None):
        value = self.scale.get()
        if int(value) != value:
            self.scale.set(round(value))

    def reset_scale(self):
        self.scale.set(256)


if __name__ == '__main__':
    app = CrackDetectionApp()
    app.run()
