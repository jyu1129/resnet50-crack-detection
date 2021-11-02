import json
import math
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from crack_detection.data_processor import DataLoader
from crack_detection.model import Model


def display_batch_of_images(directory_iterator, classes, predictions=None, name='Figure', red=False):
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
            if label is None:
                title, correct = classes[predictions[i]], True
            correct = (int(label) == predictions[i])
            title, correct = "{} [{}{}{}]".format(classes[int(label)], 'OK' if correct else 'NO',
                                                  u"\u2192" if not correct else '',
                                                  classes[predictions[i]] if not correct else ''), correct

        dynamic_title_size = figs * spacing / max(rows,
                                                  cols) * 40 + 3  # magic formula tested to work from 1x1 to 10x10

        plt.subplot(*subplot)
        plt.axis('off')
        plt.imshow(image.astype('uint8'))
        if len(title) > 0:
            plt.title(title, fontsize=int(dynamic_title_size) if not red else int(dynamic_title_size / 1.2),
                      color='red' if red else 'black',
                      fontdict={'verticalalignment': 'center'}, pad=int(dynamic_title_size / 1.5))

        subplot = subplot[0], subplot[1], subplot[2] + 1

    # layout

    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=spacing, hspace=spacing)
    plt.show()


def display_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()


def main():
    with tf.device('/device:GPU:0'):
        configs = json.load(open('config.json', 'r'))

        if configs['mode'] == 'train':
            if not os.path.exists(configs['model']['save_dir']):
                os.makedirs(configs['model']['save_dir'])

            data = DataLoader()
            ds_train = data.get_train_data(configs)
            ds_valid = data.get_validation_data(configs)
            model = Model()
            model.build_model(configs)
            steps_per_epoch = data.get_train_length() // configs['data']['batch_size']
            valid_steps = data.get_validation_length() // configs['data']['batch_size']
            model.train(
                train=ds_train,
                valid=ds_valid,
                steps_per_epoch=steps_per_epoch,
                epochs=configs['model']['epochs'],
                valid_steps=valid_steps,
                batch_size=configs['data']['batch_size'],
                save_dir=configs['model']['save_dir'],
            )
        elif configs['mode'] == 'predict-test':
            data = DataLoader()
            model = Model()
            model.load_model(configs['model']['load_dir'])
            ds_test = data.get_test_data(configs)
            predictions = model.predict_by_batch(ds_test)
            display_batch_of_images(ds_test, configs['data']['classes'], predictions)
        elif configs['mode'] == 'predict':
            model = Model()
            model.load_model(configs['model']['load_dir'])
            predictions, crack_status = model.predict_on_crops(
                configs['test_whole_image']['filepath'],
                configs['data']['classes'],
                height=configs['test_whole_image']['detect_size'][0],
                width=configs['test_whole_image']['detect_size'][1]
            )
            display_image(predictions)


if __name__ == '__main__':
    main()
