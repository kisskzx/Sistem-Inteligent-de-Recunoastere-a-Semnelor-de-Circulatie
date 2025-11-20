import os
import cv2

def load_images_from_folder(folder):
    images = []
    labels = []
    paths = []

    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".ppm", ".png", ".jpg", ".jpeg")):
                path = os.path.join(root, f)
                img = cv2.imread(path)

                if img is None:
                    print("Imagine coruptă ignorată:", path)
                    continue

                label = os.path.basename(root)
                labels.append(int(label))
                images.append(img)
                paths.append(path)

    return images, labels, paths
