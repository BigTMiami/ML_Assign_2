import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import seaborn as sns
import numpy as np


def title_to_filename(title):
    safe_title = title.replace(" ", "_")
    safe_title = safe_title.replace(":", "_")
    safe_title = safe_title.replace(",", "_")
    return f"Document/figures/working/{safe_title}.png"


def save_to_file(plt, title):
    filename = title_to_filename(title)
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(fname=filename, bbox_inches="tight")


def random_hill_chart(values, title="FILL IN"):
    fig, ax = plt.subplots(1, figsize=(4, 5))
    alogrithm_type = "Random Hill Chart"
    fig.suptitle(alogrithm_type, fontsize=16)
    ax.set_title(title)
    x = [value[0] for value in values]
    y = [value[1] for value in values]
    ax.plot(x, y)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness")

    save_to_file(plt, alogrithm_type + " " + title)


def random_hill_chart_boxplot(values, title="FILL IN"):
    fig, ax = plt.subplots(1, figsize=(4, 5))
    alogrithm_type = "Random Hill Chart"
    fig.suptitle(alogrithm_type, fontsize=16)
    ax.set_title(title)
    x = [value[2] for value in values]
    ax.boxplot(x)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness")

    save_to_file(plt, alogrithm_type + " " + title)


def random_hill_chart_seaborn_boxplot(values, title="FILL IN"):
    fig, ax = plt.subplots(1, figsize=(4, 5))
    alogrithm_type = "Random Hill Chart"
    fig.suptitle(alogrithm_type, fontsize=16)
    ax.set_title(title)
    x = [value[0] for value in values]
    print(x)
    y = np.array([value[2] for value in values])
    print(y.shape)
    print(y[0])
    # sns.boxplot(data=y, ax=ax, flierprops = dict(markerfacecolor = '0.50', markersize = .1))
    sns.lineplot(data=y, ax=ax)
    ax.set_xlabel("Iterations")
    ax.set_xticks(np.arange(10, len(x), 10))
    ax.set_ylabel("Fitness")

    save_to_file(plt, alogrithm_type + " " + title)


def random_hill_chart_lineplot(values, title="FILL IN", maximize=True):
    fig, ax = plt.subplots(1, figsize=(4, 5))
    alogrithm_type = "Random Hill Chart"
    fig.suptitle(alogrithm_type, fontsize=16)
    ax.set_title(title)
    x_axis = []
    y_axis = []
    for value in values:
        x = value[0]
        for y in value[2]:
            x_axis.append(x)

            y_axis.append(y if maximize else -y)
    sns.lineplot(x=x_axis, y=y_axis, ax=ax)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness")

    save_to_file(plt, alogrithm_type + " " + title + " lineplot")


def random_hill_chart_heatmap(values, title="FILL IN", maximize=True):
    fig, ax = plt.subplots(1, figsize=(4, 5))
    alogrithm_type = "Random Hill Chart"
    fig.suptitle(alogrithm_type, fontsize=16)
    ax.set_title(title)
    data = np.array([values[0][3]])
    for value in values:
        data = np.append(data, [value[3]], axis=0)
    data = data.transpose()
    if maximize:
        data = np.flip(data, axis=0)
    y_ticks = ["90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%", "0%"]
    sns.heatmap(data=data, ax=ax, yticklabels=y_ticks)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness Percentile vs Maximum")
    # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    # ytick_formats = mtick.FormatStrFormatter(fmt)
    # ax.yaxis.set_major_formatter(ytick_formats)
    plt.yticks(rotation=0)

    save_to_file(plt, alogrithm_type + " " + title + " heatmap")
