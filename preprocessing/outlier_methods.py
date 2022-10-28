import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def detect_outliers(df, col, method, percentile_threshold=0.01):
    if method == "IQR":  # if skewed distribution
        Q1, Q3 = np.percentile(df[col], [25, 75])
        IQR = Q3 - Q1
        ulim = Q3 + 1.5 * IQR
        llim = Q1 - 1.5 * IQR
        outliers = df[(df[col] > ulim) | (df[col] < llim)].index

    elif method == "normal_distr":
        ulim = df[col].mean() + 3 * df[col].std()
        llim = df[col].mean() - 3 * df[col].std()
        outliers = df[(df[col] > ulim) | (df[col] < llim)].index

    elif method == "percentile":
        ulim = df[col].quantile(1 - percentile_threshold)
        llim = df[col].quantile(percentile_threshold)
        outliers = df[(df[col] > ulim) | (df[col] < llim)].index

    return outliers


def plot_distribution(df, cols, plot="boxplot", ncols=4, figsize=(12, 6)):
    fig, axs = plt.subplots(int(np.ceil(len(cols) / ncols)), ncols, figsize=figsize)
    for i, col in enumerate(cols):
        if plot == "boxplot":
            sns.boxplot(df[col], fliersize=3, ax=axs[int(i / ncols), i % ncols])
        elif plot == "histogram":
            sns.histplot(
                df[col],
                ax=axs[int(i / ncols), i % ncols],
            )
    plt.tight_layout()
