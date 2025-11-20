import os, cv2, shutil
from load_data import load_images_from_folder
from clean_data import clean_images
from transform import preprocess_image
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROC_DIR = "data/processed/images_28x28_gray"
TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
TEST_DIR = "data/test"

def ensure_dirs():
    for d in [PROC_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

def save_processed(images, labels, paths):
    ensure_dirs()
    for img, lab, p in zip(images, labels, paths):
        img_proc, img_to_save = preprocess_image(img, size=28)

        out_dir = os.path.join(PROC_DIR, str(lab))
        os.makedirs(out_dir, exist_ok=True)

        fname = os.path.splitext(os.path.basename(p))[0] + ".png"
        cv2.imwrite(os.path.join(out_dir, fname), img_to_save)

def split_and_copy():
    X, y = [], []
    for lab in sorted(os.listdir(PROC_DIR)):
        labdir = os.path.join(PROC_DIR, lab)
        for f in os.listdir(labdir):
            X.append(os.path.join(labdir, f))
            y.append(int(lab))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    def cp(files, labels, dest):
        for p, lab in zip(files, labels):
            d = os.path.join(dest, str(lab))
            os.makedirs(d, exist_ok=True)
            shutil.copy(p, d)

    cp(X_train, y_train, TRAIN_DIR)
    cp(X_val, y_val, VAL_DIR)
    cp(X_test, y_test, TEST_DIR)

if __name__ == "__main__":
    images, labels, paths = load_images_from_folder(RAW_DIR)
    images, labels, paths = clean_images(images, labels, paths)
    save_processed(images, labels, paths)
    split_and_copy()
    print("Preprocesare completă ✔ Imaginile sunt acum grayscale și 28×28 px")

