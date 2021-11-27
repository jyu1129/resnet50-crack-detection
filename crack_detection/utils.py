import datetime as dt
import time
from threading import Thread
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

EPOCHS = 1
BATCH_SIZE = 64


class Timer:

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))


def display_image(image, title, subplot, red=False, title_size=16.0):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image.astype('uint8'))
    if len(title) > 0:
        plt.title(title, fontsize=int(title_size) if not red else int(title_size / 1.2),
                  color='red' if red else 'black',
                  fontdict={'verticalalignment': 'center'}, pad=int(title_size / 1.5))
    return subplot[0], subplot[1], subplot[2] + 1


def title_from_label_and_target(label, correct_label, classes):
    if correct_label is None:
        return classes[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(classes[int(label)], 'OK' if correct else 'NO',
                                u"\u2192" if not correct else '',
                                classes[correct_label] if not correct else ''), correct


def display_batch_of_images(directory_iterator, classes, predictions=None, name='Figure'):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data1
    classes = classes
    images, labels = directory_iterator.next()
    labels = np.argmax(labels, axis=-1)
    if labels is None:
        labels = [None for _ in enumerate(images)]

    # auto-squaring: this will drop data1 that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # size and spacing
    figs = 13.0
    spacing = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(num=name, figsize=(figs, figs / cols * rows))
    else:
        plt.figure(num=name, figsize=(figs / rows * cols, figs))

    # display
    for i, (image, label) in enumerate(zip(images[:rows * cols], labels[:rows * cols])):
        title = '' if label is None else classes[int(label)]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], int(label))
        dynamic_title_size = figs * spacing / max(rows,
                                                  cols) * 40 + 3  # magic formula tested to work from 1x1 to 10x10
        # images
        subplot = display_image(image, title, subplot, not correct, title_size=dynamic_title_size)

    # layout

    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=spacing, hspace=spacing)
    plt.show()


class ThreadedCamera(object):
    def __init__(self, url=0):
        self.status = None
        self.frame = None
        self.capture = cv2.VideoCapture(url)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.fps = 1/30
        self.fps_ms = int(self.fps * 1000)

        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(self.fps)

    def get_frame(self):
        return self.frame

    def x_y_shape(self):
        x_shape = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return x_shape, y_shape
