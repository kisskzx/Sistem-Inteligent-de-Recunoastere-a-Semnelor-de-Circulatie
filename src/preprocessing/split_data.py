from sklearn.model_selection import train_test_split

def split_dataset(images, labels):
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.30, stratify=labels, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

