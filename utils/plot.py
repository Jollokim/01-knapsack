import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_run(data: np.ndarray, y_label: str, save_path: str=None):
    plt.figure()
    plt.plot(data)
    plt.ylabel(y_label)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()