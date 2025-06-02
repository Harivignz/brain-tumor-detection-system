import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

# === Get Absolute Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tumor_model_path = os.path.join(BASE_DIR, 'models', 'tumor_model.h5')
unet_model_path = os.path.join(BASE_DIR, 'models', 'unet_model.h5')
stage_model_path = os.path.join(BASE_DIR, 'models', 'stage_model.h5')

# === Load Models ===
tumor_model = load_model(tumor_model_path)
unet_model = load_model(unet_model_path)
stage_model = load_model(stage_model_path)

# === Image Preprocessing Functions ===
def preprocess_image_for_classification(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Adjust as per model input
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_image_for_segmentation(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))  # Adjust based on U-Net input
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === File Dialog to Choose MRI Image ===
Tk().withdraw()
image_path = filedialog.askopenfilename(title='Select MRI Image')

if not image_path:
    print("No file selected.")
    exit()

# === Step 1: Tumor Classification ===
img_cls = preprocess_image_for_classification(image_path)
prediction = tumor_model.predict(img_cls)
tumor_detected = prediction[0][0] > 0.5

if tumor_detected:
    print("Tumor Detected ✅")

    # === Step 2: Segmentation ===
    img_seg = preprocess_image_for_segmentation(image_path)
    mask = unet_model.predict(img_seg)[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (512, 512))

    original = cv2.imread(image_path)
    original = cv2.resize(original, (512, 512))
    segmented = cv2.addWeighted(original, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # === Step 3: Stage Classification ===
    stage_input = cv2.resize(original, (128, 128))
    stage_input = stage_input.astype('float32') / 255.0
    stage_input = np.expand_dims(stage_input, axis=0)
    stage_pred = stage_model.predict(stage_input)
    stage = np.argmax(stage_pred)

    stage_text = f"Stage Detected: Grade {stage + 1}"
    print(stage_text)

    # === Display Final Results ===
    cv2.putText(segmented, stage_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    show_image("Tumor Segmentation & Stage", segmented)

else:
    print("No Tumor Detected ❌")
    show_image("MRI Image", cv2.imread(image_path))