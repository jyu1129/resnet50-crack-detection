import tensorflow as tf
import numpy as np
import os
import datetime as dt
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from urllib.request import urlopen
from crack_detection.utils import Timer


class Model:
    """A class for an building and inferencing an resnet model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5,
            decay_steps=10000,
            decay_rate=0.9)

        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        base_model.trainable = False

        self.model.add(base_model)

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler, epsilon=0.001),
                           loss=configs['model']['loss'],
                           metrics=configs['model']['metrics'])

        print('[Model] Model Compiled')
        print('[Model] Model Summary:')
        self.model.summary()

        timer.stop()

    def train(self, train, valid, steps_per_epoch, epochs, valid_steps, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            train,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=valid,
            validation_steps=valid_steps,
            callbacks=[callbacks]
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict_by_batch(self, data):
        # Predict batch of images
        print('[Model] Predicting batch-by-batch...')
        step_size_test = data.n // data.batch_size
        data.reset()
        probabilities = self.model.predict(data, steps=step_size_test, verbose=1)
        predictions = np.argmax(probabilities, axis=-1)

        return predictions
        # display_batch_of_images(data, self.classes, predictions)

    def predict_on_crops(self, input_image, classes, https=False, height=256, width=256, save_crops=False):
        # Run prediction on whole image
        if https:
            req = urlopen(input_image)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            im = cv2.imdecode(arr, -1)
        else:
            im = cv2.imread(input_image)

        if len(im.shape) == 3:
            img_height, img_width, channels = im.shape
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            img_height, img_width, channels = im.shape
        k = 0
        output_image = np.zeros_like(im)
        crack_status = False
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        for i in range(0, img_height, height):
            for j in range(0, img_width, width):
                a = im[i:i + height, j:j + width]
                a = np.expand_dims(a, axis=0)
                processed_a = test_datagen.flow(a).next()
                # discard image crops that are not full size
                predicted_class = classes[int(np.argmax(self.model.predict(processed_a), axis=-1))]
                # # save image
                # file, ext = os.path.splitext(input_image)
                # image_name = file.split('/')[-1]
                # folder_name = 'out_' + image_name
                # Put predicted class on the image
                if predicted_class == 'crack':
                    color = (0, 0, 255)
                    crack_status = True
                else:
                    color = (0, 255, 0)
                cv2.putText(a, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
                b = np.zeros_like(a, dtype=np.uint8)
                b[:] = color
                add_img = cv2.addWeighted(a, 0.9, b, 0.1, 0, dtype=cv2.CV_64F)
                # Save crops
                # if save_crops:
                #     if not os.path.exists(os.path.join('predictions', folder_name)):
                #         os.makedirs(os.path.join('predictions', folder_name))
                #     filename = os.path.join('predictions', folder_name, 'img_{}.png'.format(k))
                #     cv2.imwrite(filename, add_img)
                output_image[i:i + height, j:j + width, :] = add_img
                k += 1
        # # Save output image
        # cv2.imwrite(os.path.join('predictions', folder_name + '.jpg'), output_image)

        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        return output_image, crack_status
