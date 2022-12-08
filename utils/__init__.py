import numpy as np

def balance(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, classes_samples = np.unique(Y, return_counts=True)
    smallest_class = classes[np.argsort(classes_samples)][0]
    samples = classes_samples.min()
    X_list, Y_list = list(), list()
    stat = {class_: 0 for class_ in classes}

    for x, y in zip(X, Y):
        if y != smallest_class and stat[y] >= samples:
            continue
        else:
            Y_list.append(y)
            X_list.append(x)
            stat[y] += 1

    return np.array(X_list), np.array(Y_list)
