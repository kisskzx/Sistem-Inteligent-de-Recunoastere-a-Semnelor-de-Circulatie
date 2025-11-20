import cv2
import numpy as np

def preprocess_image(img, size=28):
    # Conversie în grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensionare
    img_resized = cv2.resize(img_gray, (size, size))

    # Normalizare [0, 1]
    img_norm = img_resized.astype("float32") / 255.0

    # Pentru salvare ca imagine PNG
    img_to_save = (img_norm * 255).astype("uint8")

    return img_norm, img_to_save
