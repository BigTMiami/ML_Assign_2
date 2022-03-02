import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd


def title_to_filename(title):
    safe_title = title.replace(" ", "_")
    safe_title = safe_title.replace(":", "_")
    safe_title = safe_title.replace(",", "_")
    safe_title = safe_title.replace("[", "")
    safe_title = safe_title.replace("]", "")
    safe_title = safe_title.replace("(", "")
    safe_title = safe_title.replace(")", "")
    safe_title = safe_title.replace("'", "")
    return f"Document/figures/working/{safe_title}.png"


def save_to_file(plt, title):
    filename = title_to_filename(title)
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(fname=filename, bbox_inches="tight")


def dict_to_str(values):
    dict_string = ""
    for key, value in values.items():
        dict_string += f"_{key}_{value}"
    dict_string += "_"
    return dict_string


def fitness_chart(
    df, line_col, title="TITLE", sup_title="SUPTITLE", maximize=True, xscale_log=False, info_settings={}
):
    df_max = df.groupby(["Iteration", line_col]).agg({"Fitness": "max"}).reset_index()
    color_count = len(pd.unique(df[line_col]))
    palette = sns.color_palette("hls", color_count)
    fig, ax = plt.subplots(1, figsize=(4, 5))
    if not maximize:
        ax.invert_yaxis()
    if xscale_log:
        ax.set_xscale("log")
    fig.suptitle(sup_title, fontsize=16)
    ax.set_title(title)
    sns.lineplot(data=df_max, x="Iteration", y="Fitness", hue=line_col, palette=palette, ax=ax)
    info_settings_str = dict_to_str(info_settings)
    log_tag = "_log" if xscale_log else ""
    save_to_file(plt, sup_title + " " + title + info_settings_str + log_tag)


def time_chart(algorithms, times, title="TITLE", sup_title="SUPTITLE", info_settings={}):

    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(sup_title, fontsize=16)
    ax.set_title(title)
    sns.barplot(x=algorithms, y=times, ax=ax)
    ax.set_yscale("log")
    plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large")
    ax.set_ylabel("Time (s)")
    info_settings_str = dict_to_str(info_settings)
    save_to_file(plt, sup_title + " " + title + info_settings_str)


def neural_training_chart(scores, start_node=0, title="TITLE", sup_title="SUPTITLE", chart_loss=False):
    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(title, fontsize=16)
    x = list(range(start_node, len(scores)))
    ax.set_title(sup_title)
    if chart_loss:
        ax.plot(x, scores[start_node:, 1], label="Training Loss")
        ax.set_ylabel("Loss")
    else:
        ax.plot(x, 100.0 - scores[start_node:, 2], label="Training Data")
        ax.plot(x, 100.0 - scores[start_node:, 4], label="Test Data")
        ax.set_ylabel("Error")
    ax.set_xlabel("Epochs")

    plt.legend()

    save_to_file(plt, sup_title + " " + title)
