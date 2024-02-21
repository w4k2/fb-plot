import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from config import *

from scipy.stats import gaussian_kde, ttest_rel

RESULTS = os.listdir("results")

CLF = 0

clf_names = [s.__name__ for s in SAMPLING]

# betas = np.hstack([np.linspace(0.1, 1.0, 1000), np.linspace(1.0, 10, 1000)])
betas = np.geomspace(0.1, 10, 100)

def f_beta(TPR, PPV, beta=1.0):
    if not hasattr(beta, "__iter__"):
        beta = np.array([beta])

    beta_sqr = beta ** 2

    return np.nan_to_num(
        (1 + beta_sqr)
        * PPV[..., np.newaxis]
        * TPR[..., np.newaxis]
        / ((beta_sqr * PPV[..., np.newaxis]) + TPR[..., np.newaxis])
    )


for res_file in RESULTS:
    print(res_file)
    results = np.load(os.path.join("results", res_file))
    ds_name = res_file.split('.')[0]

    n_samplers = results.shape[0]
    
    data = results[:, CLF, :, :]

    recall = data[:, :, 0]
    precision = data[:, :, 1]
    n_splits = data.shape[1]

    f_betas = f_beta(recall, precision, beta=betas)

    f_betas_mean = f_betas.mean(axis=1)
    f_betas_std = f_betas.std(axis=1)

    best = np.argmax(f_betas_mean, axis=0)

    u, ind = np.unique(best, return_index=True)
    unique_best = u[np.argsort(ind)]

    p_values = np.zeros(shape=(len(betas), n_samplers))

    # for ai, best_a in enumerate(unique_best):
    for ai in range(n_samplers):
        for i in range(len(betas)):
            p_values[i, ai] = ttest_rel(f_betas[best[i], :, i], f_betas[ai, :, i]).pvalue

    stat_best = np.all(np.nan_to_num(p_values, 0) < 0.05, axis=1)
    stat_c_point = np.argwhere(np.diff(stat_best)).ravel()

    fig, axs = plt.subplots(1, 2, figsize=(4 * 3, 4.3), gridspec_kw={"width_ratios": [1, 2]})

    ax = axs[0]

    # gs = gridspec.GridSpecFromSubplotSpec(
    #     2, 2, height_ratios=(1, 12), width_ratios=(12, 1), subplot_spec=ax
    # )

    # ax.remove()

    # ax = fig.add_subplot(gs[1, 0])
    # ax_x = fig.add_subplot(gs[1, 1], sharey=ax)
    # ax_y = fig.add_subplot(gs[0, 0], sharex=ax)

    ax.grid(ls=":")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("TPR")
    ax.set_ylabel("PPV")
    ax.spines[["right", "top"]].set_visible(False)

    # ax_y.xaxis.grid(True, ls=":")
    # ax_y.spines[["right", "top", "left"]].set_visible(False)
    # ax_y.set_xlim(-0.05, 1.05)
    # ax_y.set_ylim(-0.05, 1.05)
    # ax_y.tick_params(axis=u'y', which=u'both',length=0)
    # plt.setp(ax_y.get_xticklabels(), visible=False)
    # plt.setp(ax_y.get_yticklabels(), visible=False)

    # ax_x.yaxis.grid(True, ls=":")
    # ax_x.spines[["right", "top", "bottom"]].set_visible(False)
    # ax_x.set_xlim(-0.05, 1.05)
    # ax_x.set_ylim(-0.05, 1.05)
    # ax_x.tick_params(axis=u'x', which=u'both',length=0)
    # plt.setp(ax_x.get_yticklabels(), visible=False)
    # plt.setp(ax_x.get_xticklabels(), visible=False)

    kde_space = np.linspace(0, 1, 200)

    ax.scatter(recall.ravel(), precision.ravel(), s=40, c="#DDDDDD")

    for b in unique_best:
        ax.scatter(recall[b], precision[b], s=80, marker="D", edgecolors='k', facecolor='none')
        sc = ax.scatter(recall[b], precision[b], s=50, marker="D", edgecolor='w', alpha=0.7)

        # try:
        #     recall_kde = gaussian_kde(recall[b].T)(kde_space)
        #     recall_kde /= np.max(recall_kde)
        #     precision_kde = gaussian_kde(precision[b].T)(kde_space)
        #     precision_kde /= np.max(precision_kde)
        # except:
        #     continue

        # color = sc.get_facecolors()[0].tolist()
        # color_alpha = lambda b: np.array([*color[:3], b])

        # ax_y.plot(kde_space, recall_kde, lw=1.2)
        # ax_x.plot(precision_kde, kde_space, lw=1.2)

    ax = axs[1]

    ax.grid(ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.00)
    ax.set_xlim(np.min(betas), np.max(betas))
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel("$F_\\beta$ score")
    ax.set_xlabel("$\\beta$")

    stat_line = np.repeat(np.nan, len(stat_best))
    stat_line[:-1][np.logical_or(stat_best[:-1], np.diff(stat_best))] = 0.0
    stat_line[-1] = stat_line[-2]
    ax.plot(betas, stat_line, color='k', lw=10)

    # for b in range(len(SAMPLING)):
    #     ax.plot(betas, f_betas_mean[b], c="#CCCCCC", lw=1)
    #     ax.fill_between(
    #         betas,
    #         f_betas_mean[b] + f_betas_std[b],
    #         f_betas_mean[b] - f_betas_std[b],
    #         alpha=0.01,
    #         color='#CCCCCC'
    #     )

    changing_points = []

    for b in unique_best:
        best_map = (best == b)
        step_before = np.argwhere(best_map)[0][0] - 1
        if step_before != -1:
            best_map[step_before] = True
            changing_points.append(step_before)

        p = ax.plot(betas[best_map], f_betas_mean[b][best_map], label=clf_names[b], lw=3)

        ax.fill_between(
            betas[best_map],
            f_betas_mean[b][best_map] + f_betas_std[b][best_map],
            f_betas_mean[b][best_map] - f_betas_std[b][best_map],
            alpha=0.2,
        )

        ax.plot(
            betas, f_betas_mean[b] + f_betas_std[b], c=p[0].get_color(), alpha=0.1, ls="-"
        )
        ax.plot(
            betas, f_betas_mean[b] - f_betas_std[b], c=p[0].get_color(), alpha=0.1, ls="-"
        )

    # ax.legend(loc="upper center", prop={"size": "x-small"}, ncol=8)
    ax.legend(loc='lower left', bbox_to_anchor=(0.025, 0.025), fancybox=False, ncol=2, prop={'size': 'large'})


    for i in np.array(changing_points):
        a = betas[i]
        ax.vlines(a, 0.0, 1.00, color="k", lw=1, ls=":")

        ax.text(
            a,
            0.92,
            f"{a:.2f}",
            rotation=90,
            fontsize="smaller",
            horizontalalignment="right",
        )

    for i in stat_c_point:
        a = betas[i]

        ax.text(
            a,
            0.025,
            f"{a:.2f}",
            rotation=00,
            fontsize="smaller",
            horizontalalignment="center",
        )

    fig.tight_layout()
    plt.savefig(f"figures/04_{ds_name}.png")
    plt.savefig(f"figures/04_{ds_name}.pdf")
    plt.close()
