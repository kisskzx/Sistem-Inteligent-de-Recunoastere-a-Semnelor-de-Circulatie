def clean_images(images, labels, paths):
    clean_imgs = []
    clean_lbls = []
    clean_paths = []

    for img, lbl, p in zip(images, labels, paths):
        if img is not None and img.size > 0:
            clean_imgs.append(img)
            clean_lbls.append(lbl)
            clean_paths.append(p)

    return clean_imgs, clean_lbls, clean_paths

