"""
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization for the
ResNet50 crack detection model.

Grad-CAM highlights which regions of an image the model focused on when
making its prediction — turning a black-box decision into something you
can actually see and trust.

Usage:
    python -m crack_detection.gradcam <model_path> <image_path> [output_path]

Example:
    python -m crack_detection.gradcam models/26102021-231847-e100.h5 my_wall.jpg
"""

import sys
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


LAST_CONV_LAYER = 'conv5_block3_out'
CLASSES = ['crack', 'non crack']
INPUT_SIZE = (256, 256)


def _build_grad_model(model):
    """
    Constructs a model that outputs both:
      - the activations of the last ResNet50 conv layer
      - the final softmax predictions
    """
    base_model = model.layers[0]  # ResNet50 is the first layer of the Sequential
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[base_model.get_layer(LAST_CONV_LAYER).output, model.output]
    )


def compute_heatmap(img_array, grad_model, class_index=None):
    """
    Runs a forward + backward pass to get the Grad-CAM heatmap.

    Args:
        img_array: preprocessed image array, shape (1, H, W, 3)
        grad_model: model with two outputs (conv activations, predictions)
        class_index: which class to explain; defaults to the predicted class

    Returns:
        heatmap (H x W numpy array, values in [0, 1])
        predictions (raw softmax output)
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalise to [0, 1]; avoid division by zero on flat regions
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), predictions.numpy()


def overlay_heatmap(original_rgb, heatmap, alpha=0.45):
    """
    Resizes the heatmap to match the image and blends them together.

    Returns:
        colormap  — the raw JET heatmap as an RGB image
        overlay   — the heatmap blended onto the original image
    """
    h, w = original_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    colormap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb, 1 - alpha, colormap, alpha, 0)
    return colormap, overlay


def run_gradcam(model_path, image_path, output_path='reports/figures/gradcam_output.png'):
    """
    Full pipeline: load model → preprocess image → compute Grad-CAM → save figure.

    Args:
        model_path:  path to the saved .h5 model
        image_path:  path to the input image (jpg/png)
        output_path: where to save the output figure

    Returns:
        (predicted_class: str, confidence: float)
    """
    print(f'[GradCAM] Loading model: {model_path}')
    model = load_model(model_path)
    grad_model = _build_grad_model(model)

    print(f'[GradCAM] Reading image: {image_path}')
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f'Could not read image at: {image_path}')

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, INPUT_SIZE)

    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_preprocessed = preprocess_input(img_array.copy())

    heatmap, preds = compute_heatmap(img_preprocessed, grad_model)
    pred_index = int(np.argmax(preds[0]))
    pred_class = CLASSES[pred_index]
    confidence = float(np.max(preds[0])) * 100

    print(f'[GradCAM] Prediction: {pred_class.upper()} — {confidence:.1f}% confidence')

    colormap, overlay = overlay_heatmap(img_resized, heatmap)

    # ── Build figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0d1117')

    panel_titles = [
        'Original Image',
        'Grad-CAM Heatmap\n(where the model looks)',
        f'Result: {pred_class.upper()}\nConfidence: {confidence:.1f}%',
    ]
    panel_images = [img_resized, colormap, overlay]
    result_color = '#ff5555' if pred_class == 'crack' else '#50fa7b'
    title_colors = ['#f8f8f2', '#f8f8f2', result_color]

    for ax, title, img, color in zip(axes, panel_titles, panel_images, title_colors):
        ax.imshow(img)
        ax.set_title(title, color=color, fontsize=13, fontweight='bold', pad=10)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle(
        'Explainable AI — Grad-CAM Crack Detection',
        color='#bd93f9', fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    print(f'[GradCAM] Saved → {output_path}')
    return pred_class, confidence


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    _model_path = sys.argv[1]
    _image_path = sys.argv[2]
    _output_path = sys.argv[3] if len(sys.argv) > 3 else 'reports/figures/gradcam_output.png'

    run_gradcam(_model_path, _image_path, _output_path)
