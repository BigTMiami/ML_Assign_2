import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import seaborn as sns
import numpy as np
import pandas as pd
import re


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


def fitness_chart(df, line_col, title="TITLE", sup_title="SUPTITLE", info_settings={}):
    df_max = df.groupby(["Iteration", line_col]).agg({"Fitness": "max"}).reset_index()
    color_count = len(pd.unique(df[line_col]))
    palette = sns.color_palette("hls", color_count)
    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(sup_title, fontsize=16)
    ax.set_title(title)
    sns.lineplot(data=df_max, x="Iteration", y="Fitness", hue=line_col, palette=palette, ax=ax)
    info_settings_str = dict_to_str(info_settings)
    save_to_file(plt, sup_title + " " + title + info_settings_str)


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
