import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import csv
from matplotlib.pyplot import MultipleLocator
SIZE = 13
sns.set_style("ticks")


def smoother(x, a=0.9, w=10, mode="moving"):
    if mode is "moving":
        y = [x[0]]
        for i in range(1, len(x)):
            y.append((1 - a) * x[i] + a * y[i - 1])
    elif mode is "window":
        y = []
        for i in range(len(x)):
            y.append(np.mean(x[max(i - w, 0):i + 1]))
    return y


def plot_single(df, title=None):
    x = df["step"].to_numpy() / 1e6
    y = smoother(df["reward"].to_numpy(), w=10, mode="window")
    plt.xlabel("Million Steps")
    plt.ylabel("Average Return")
    if title:
        plt.title(title)
    plt.plot(x, y)


def plot_multi(dfs, labels, title=None):
    plt.figure(figsize=(8, 6), dpi=100)
    for df, label in zip(dfs, labels):
        x = df["exploration/num steps total"].to_numpy() / 1e6
        y = smoother(df["maxtime"].to_numpy(), w=10, mode="window")
        plt.plot(x, y, label=label)
    plt.xlabel("Million Steps")
    plt.ylabel("Average Return")
    if title:
        plt.title(title)
    plt.legend()


def plot_curve(ax, dfs, label=None, color=sns.color_palette()[0], shaded_err=False, shaded_std=True, shared_area=0.5, SEAC=False):
    # here SEAC calculate sum rewards but ours calculate mean rewards
    print(label)
    length = min([df.to_numpy().shape[0] for df in dfs]) - 1
    N = length
    x = dfs[0]["step"].to_numpy()[:N] / 1e6
    print(x.shape)
    #     ys = [smoother(df["evaluation/Average Returns"].to_numpy()[:N], w=20, mode="window") for df in dfs]
    if SEAC:
        ys = [smoother(df["reward"].to_numpy()[:N] / 4, a=0.9) for df in dfs]
    else:
        try:
            ys = [smoother(100 * df["win_rate"].to_numpy()[:N], a=0.9)
                  for df in dfs]
        except:
            ys = [smoother(df["reward"].to_numpy()[:N] + 1, a=0.9)
                  for df in dfs]
    #     ys = [smoother(df["trainer/Log Pis Mean"].to_numpy()[:N:10], a=0.9) for df in dfs]
    y_mean = np.mean(ys, axis=0)
    y_mean = np.mean(ys, axis=0)
    if label is None:
        lin = ax.plot(x, y_mean, color=color, linewidth=3)
        # plt.semilogy(x, y_mean, color=color)
    else:
        lin = ax.plot(x, y_mean, color=color, label=label, linewidth=1.8)

    if len(ys) > 0:
        y_std = np.std(ys, axis=0)
        ax.fill_between(x, (y_mean-y_std).clip(min=0), y_mean +
                        y_std, color=color, alpha=.2)

    return lin


def plot_w(ax, dfs, label=None, color=sns.color_palette()[0], shaded_err=False, shaded_std=True, shared_area=0.5):
    #     ys = [smoother(df["evaluation/Average Returns"].to_numpy()[:N], w=20, mode="window") for df in dfs]
    ys = [smoother(np.array(df), a=0.9) for df in dfs]
    x = np.array(range(min([np.size(y) for y in ys]))) + 1
    x = 5000 * x / 1e6
    #     ys = [smoother(df["trainer/Log Pis Mean"].to_numpy()[:N:10], a=0.9) for df in dfs]
    y_mean = np.mean(ys, axis=0)
    if label is None:
        lin = ax.plot(x, y_mean, color=color)
        # plt.semilogy(x, y_mean, color=color)
    else:
        lin = ax.plot(x, y_mean, color=color, label=label)
    if len(ys) > 1:
        y_std = np.std(ys, axis=0) * shared_area
        y_stderr = y_std / np.sqrt(len(ys))
        if shaded_err:
            ax.fill_between(x, y_mean - y_stderr, y_mean +
                            y_stderr, color=color, alpha=.4)
        if shaded_std:
            ax.fill_between(x, y_mean - y_std, y_mean +
                            y_std, color=color, alpha=.2)

    return lin


def getdata(path):
    list_r = []
    with open(path, "rt") as f:
        csv_read = csv.reader(f)
        print(next(csv_read))
        print(csv_read)
        for line in csv_read:

            list_r.append(float(line[1][1:-1]))

    return list_r


def plot_list(ax, basedirs, subdirs, labels, title):
    for i in range(len(basedirs)):
        load_in = basedirs[i]
        #load_in = basedirs[i]
        subdirs = [item for item in os.listdir(load_in) if 'seed' in item]
        df = [pd.read_csv(load_in + '/' + j, index_col=None,
                          header=0, engine='python') for j in subdirs]

        plot_curve(ax, df, label=labels[i], color=COLORS[i + 1], SEAC=False)

    if title:
        plt.title(title)


def plot_array(ax, basedirs, subdirs, labels, title):
    for i in range(len(basedirs)):
        df = [getdata(basedirs[i] + subdirs[i] + str(j + 1) +
                      '00/reward.csv') for j in range(1, 3)]
        plot_w(ax, df, label=labels[i], color=COLORS[i])

    if title:
        plt.title(title)


def choose_list(list_c, index):
    list_o = []
    for i in index:
        list_o.append(list_c[i])

    return list_o


fig = plt.figure(10, figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)
COLORS = sns.color_palette("Set1", 50, 0.9)
#COLORS = ['black', 'darkred']

file = [item for item in os.listdir(
    './') if 'Q' in item]
# and 'alpha_0.1_class_num_2_KL_type_1_weight_0.1_con_weight_0.1' in item
start = 0

if (0 == 0):
    basedirs = [
        'CDS_QPLEX',
        'CDS_QMIX',
        'QPLEX',
        'QMIX',
    ]

    labels = [item for item in basedirs]
    '''labels = [
        'CDS+QMIX', 'CDS+QMIX+TDloss_prior (best alpha=0.2)', 'CDS+QMIX+TDloss_prior (worst alpha=0.8)']'''
    ax.tick_params(labelsize=SIZE)
    #labels = ['CDS+coach']

    #labels = ['sdiqmix', 'iqmix', 'sdiqplex', 'iqplex', 'qmix', 'qplex']
    #labels = ['sdiqmix1', 'sdiqmix2', 'sdiqmix3', 'sdiqplex1', 'sdiqplex2', 'sdiqplex3', 'qmix', 'qplex']

    title = 'academy counterattack hard'
    plot_list(ax, basedirs, None, labels, title)
    plt.ylabel('Test Win(mean)%', fontsize=SIZE)
    plt.xlabel('time(1e6)', fontsize=SIZE)
    #x_major_locator = MultipleLocator(1)
    # ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(loc="upper left", prop={'size': SIZE}, edgecolor="white")
    # plt.legend(loc="upper left", bbox_to_anchor=(-1.25, 1.0),
    #           prop={'size': SIZE}, edgecolor="white")

    sns.despine()
    fig.savefig(f'grf.pdf', dpi=300, bbox_inches='tight')

if (1 == 0):

    basedirs = [
        'sdiqmix_double_q_True_norm_weight_1.0_intrinsic_weight_2.0_0.5_0.05',
        'sdiqmix_double_q_True_norm_weight_1.0_intrinsic_weight_2.0_0.5_0.5',
        'sdiqmix_double_q_True_norm_weight_1.0_intrinsic_weight_0.5_1.0_0.5',
        'sdiqmix_double_q_True_norm_weight_1.0_intrinsic_weight_0.5_0.5_0.05',
        'sdiqmix_double_q_True_norm_weight_1.0_intrinsic_weight_1.0_1.0_0.05',
        'sdiqmix_double_q_True_norm_weight_1.0_intrinsic_weight_1.0_0.5_0.05',
        'sdiqmix_double_q_True_norm_weight_1.0_intrinsic_weight_1.0_1.0_0.5',
        'sdiqmix_double_q_True_norm_weight_1.0_intrinsic_weight_1.0_2.0_0.05',
    ]

    labels = basedirs.copy()
    labels = [
        'beta_2.0_0.5_0.05',
        'beta_2.0_0.5_0.5',
        'beta_0.5_1.0_0.05',
        'beta_0.5_0.5_0.05',
        'beta_1.0_1.0_0.05',
        'beta_1.0_0.5_0.05',
        'beta_1.0_1.0_0.5',
        'beta_1.0_2.0_0.05',
    ]
    #labels = ['sdiqmix', 'sdiqplex', 'qmix', 'qplex']
    #labels = ['sdiqmix1', 'sdiqmix2', 'sdiqmix3', 'sdiqplex1', 'sdiqplex2', 'sdiqplex3', 'qmix', 'qplex']

    title = 'SC2'
    plot_list(ax, basedirs, None, labels, title)
    plt.legend(loc="upper left", edgecolor="white")
    plt.xlabel('time(1e6)')
    plt.ylabel('win_rate')
    plt.xlim(0, 2)

    sns.despine()
    fig.savefig('fig/qmix.pdf', dpi=300, bbox_inches='tight')

if (1 == 0):

    basedirs = [
        'qplex_sdi_intrinsic_norm_weight_1.0_intrinsic_weight_0.5_0.5_0.05_anneal_linear_0.3_2000000_1000.0',
        'qplex_sdi_intrinsic_norm_weight_1.0_intrinsic_weight_0.5_0.5_0.05_anneal_linear_1.0_2000000_1000.0',
        'sdiqmix_norm_weight_1.0_intrinsic_weight_1.0_2.0_0.05_anneal_linear_0.3_4000000_1000.0',
        'sdiqmix_norm_weight_1.0_intrinsic_weight_1.0_2.0_0.05_anneal_linear_1.0_4000000_1000.0',
        'sqvdn_norm_weight_1.0_intrinsic_weight_1.0_0.5_0.05_anneal_linear_0.3_1000000_1000.0',
        'sqvdn_norm_weight_1.0_intrinsic_weight_1.0_0.5_0.05_anneal_linear_1.0_1000000_1000.0',
    ]

    labels = basedirs.copy()
    labels = [
        'qplex_intrinsic_best_0.3',
        'qplex_intrinsic_best_1.0',
        'sdqmix_intrinsic_best_0.3',
        'sdqmix_intrinsic_best_1.0',
        'sdvdn_intrinsic_best_0.3',
        'sdvdn_intrinsic_best_1.0',
    ]
    #labels = ['sdiqmix', 'sdiqplex', 'qmix', 'qplex']
    #labels = ['sdiqmix1', 'sdiqmix2', 'sdiqmix3', 'sdiqplex1', 'sdiqplex2', 'sdiqplex3', 'qmix', 'qplex']

    title = '3_vs_2'
    plot_list(ax, basedirs, None, labels, title)
    plt.legend(loc="upper left", edgecolor="white")
    plt.xlabel('time(1e6)')
    plt.ylabel('win_rate')

    sns.despine()
    fig.savefig('grf.pdf', dpi=300, bbox_inches='tight')

plt.close()
