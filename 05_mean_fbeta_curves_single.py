import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import os
from config import *

from scipy.stats import gaussian_kde, ttest_rel


CLF = 0

clf_names = [s.__name__ for s in SAMPLING]

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

n_datasets = len(DATASETS)

fig, axs = plt.subplots(n_datasets // 2, 2, figsize=(3 * 3.5, 3 * n_datasets // 2), sharex=True, sharey=True)

for res_idx, ds_name in enumerate(DATASETS):
    res_file = os.path.join("results", f"{ds_name}.npy")
    results = np.load(res_file)
    print(ds_name)
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

    for ai in range(n_samplers):
        for i in range(len(betas)):
            p_values[i, ai] = ttest_rel(f_betas[best[i], :, i], f_betas[ai, :, i]).pvalue

    stat_best = np.all(np.nan_to_num(p_values, 0) < 0.05, axis=1)
    stat_c_point = np.argwhere(np.diff(stat_best)).ravel()

    ax = axs[res_idx // 2, res_idx % 2]
    ax.set_title(ds_name)

    ax.grid(ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.00)
    ax.set_xlim(np.min(betas), np.max(betas))
    ax.spines[["right", "top"]].set_visible(False)

    stat_line = np.repeat(np.nan, len(stat_best))
    stat_line[:-1][np.logical_or(stat_best[:-1], np.diff(stat_best))] = 0.0
    stat_line[-1] = stat_line[-2]
    ax.plot(betas, stat_line, color='k', lw=10)

    for b in unique_best:
        ax.plot(betas, f_betas_mean[b], c="#CCCCCC", alpha=0.8)
        ax.fill_between(
            betas,
            f_betas_mean[b] + f_betas_std[b],
            f_betas_mean[b] - f_betas_std[b],
            alpha=0.1,
            color='#CCCCCC'
        )

    changing_points = []

    for b in unique_best:
        best_map = (best == b)
        step_before = np.argwhere(best_map)[0][0] - 1
        if step_before != -1:
            best_map[step_before] = True
            changing_points.append(step_before)

        p = ax.plot(betas[best_map], f_betas_mean[b][best_map], label=clf_names[b], lw=2)

        ax.fill_between(
            betas[best_map],
            f_betas_mean[b][best_map] + f_betas_std[b][best_map],
            f_betas_mean[b][best_map] - f_betas_std[b][best_map],
            alpha=0.15,
        )

        ax.plot(
            betas, f_betas_mean[b] + f_betas_std[b], c=p[0].get_color(), lw=1, ls="-", alpha=0.2
        )
        ax.plot(
            betas, f_betas_mean[b] - f_betas_std[b], c=p[0].get_color(), lw=1, ls="-", alpha=0.2
        )

    ax.legend(loc='lower left', bbox_to_anchor=(0.025, 0.05), fancybox=True, ncol=2, prop={'size': 'smaller'})

    for i in np.array(changing_points):
        a = betas[i]
        ax.vlines(a, 0.0, 1.00, color="k", lw=1.2, ls=":")

        ax.text(
            a,
            0.90,
            f"{a:.2f}",
            rotation=90,
            fontsize="small",
            horizontalalignment="right",
        )

    for i in stat_c_point:
        a = betas[i]

        ax.text(
            a,
            0.03,
            f"{a:.2f}",
            rotation=00,
            fontsize="small",
            horizontalalignment="center",
        )

plt.tight_layout()
plt.savefig(f"figures/all.png")
plt.savefig(f"figures/all.pdf")
plt.close()
