import numpy as np
import pandas as pd
import seaborn as sns


def plot_boxplots(ax, perf_dict_for_pd, x_label, y_label):
    perf_df = pd.DataFrame.from_dict(perf_dict_for_pd)
    our_mean_color = sns.color_palette("muted")[9]
    marker_size = 7
    mean_markers = 'X'
    with sns.color_palette("muted"):
        sns.boxplot(x=x_label, y=y_label, data=perf_df, ax=ax, showfliers=False)
        ax.plot([0], [np.mean(perf_df[y_label])], color=our_mean_color, marker=mean_markers,
                markeredgecolor='#545454', markersize=marker_size, zorder=10)


def plot_barplots(ax, perf_dict_for_pd, x_label, y_label):
    perf_df = pd.DataFrame.from_dict(perf_dict_for_pd)
    with sns.color_palette("muted"):
        sns.barplot(x=x_label, y=y_label, ax=ax, data=perf_df)