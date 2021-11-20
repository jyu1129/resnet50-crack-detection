import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import ImageTk, Image


def predict_on_crops(model, input_image, classes, height=256, width=256):
    file_bytes = np.asarray(bytearray(input_image.read()), dtype=np.uint8)
    im = cv2.imdecode(file_bytes, 1)

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
            predicted_class = classes[int(np.argmax(model.predict(processed_a), axis=-1))]
            if predicted_class == 'crack':
                color = (0, 0, 255)
                crack_status = True
            else:
                color = (0, 255, 0)
            cv2.putText(a, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
            b = np.zeros_like(a, dtype=np.uint8)
            b[:] = color
            add_img = cv2.addWeighted(a, 0.9, b, 0.1, 0, dtype=cv2.CV_64F)
            output_image[i:i + height, j:j + width, :] = add_img
            k += 1

    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_image, crack_status


with tf.device('/device:GPU:0'):
    model = load_model("models/26102021-231847-e100.h5")

st.write("""
        # Wall Crack Detection
        """)

st.write("Crack Detection is a tool that utilizes deep learning to recognize concrete wall cracks in pictures.")

file = st.file_uploader("Please upload an image file", type=["png", "jpg"])

scale = st.slider("Pick a kernel size", 128, 512, value=256)

if file is None:
    st.text("You haven't upload an image file")
else:
    # image = ImageTk.PhotoImage(file)
    st.image(file)
    with tf.device('/device:GPU:0'):
        predictions, crack_status = predict_on_crops(model, file, ['crack', 'non crack'], height=scale, width=scale)
        print(type(predictions))
        st.image(np.asarray(predictions))
        if crack_status:
            st.text('Detected!')
        else:
            st.text('Undetected')
