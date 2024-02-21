import numpy as np
import matplotlib.pyplot as plt
import os
from config import *

RESULTS = os.listdir("results")

CLF = 0

clf_names = [s.__name__ for s in SAMPLING]

# betas = np.hstack([np.linspace(0.05, 1.0, 1000), np.linspace(1.0, 50, 1000)])
betas = np.geomspace(0.1, 10, 1000)

def f_beta(TPR, PPV, beta=1.0):
    if not hasattr(beta, "__iter__"):
        beta = np.array([beta])

    beta_sqr = np.power(beta, 2)

    return np.nan_to_num(
        (1 + beta_sqr)
        * PPV[..., np.newaxis]
        * TPR[..., np.newaxis]
        / ((beta_sqr * PPV[..., np.newaxis]) + TPR[..., np.newaxis])
    )


def f_beta_zero_point(TPR_A, PPV_A, TPR_B, PPV_B):

    return np.sqrt(
        (TPR_A * TPR_B * (PPV_B - PPV_A)) / (PPV_A * PPV_B * (TPR_A - TPR_B))
    )


for res_file in RESULTS:
    results = np.load(os.path.join("results", res_file))
    ds_name = res_file.split('.')[0]
    print(ds_name)

    data = results[:, CLF, :, :]

    recall = data[:, :, 0]
    precision = data[:, :, 1]
    n_splits = data.shape[1]

    fig, axs = plt.subplots(
        n_splits, 2, figsize=(15, 3 * n_splits), gridspec_kw={"width_ratios": [1, 3]}
    )

    # Splits
    for n in range(n_splits):
        f_betas = f_beta(recall[:, n], precision[:, n], beta=betas).T
        best = np.argmax(f_betas, axis=-1)

        a = np.pad(best[1:], (0, 1), "edge") - best
        best_map = a != 0
        best_map[-1] = True
        unique_best = best[best_map]

        ax = axs[n, 0]

        ax.grid(ls=":")
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("TPR")
        ax.set_ylabel("PPV")
        ax.spines[["right", "top"]].set_visible(False)

        ax.scatter(recall[:, n], precision[:, n], s=10, c="#DDDDDD")

        for b in unique_best:
            ax.scatter(recall[b, n], precision[b, n], s=50, marker="*")

        ax = axs[n, 1]

        ax.grid(ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(np.min(betas), np.max(betas))
        ax.set_ylabel("$F_\\beta$ score")

        if n == n_splits - 1:
            ax.set_xlabel("$\\beta$")

        ax.spines[["right", "top"]].set_visible(False)

        ax.plot(betas, f_betas, c="#DDDDDD", lw=1)

        for b in unique_best:
            ax.plot(betas[best == b], f_betas[:, b][best == b], label=clf_names[b])

        ax.legend(loc="lower center", prop={"size": "x-small"}, ncol=8)

        best_iter = iter(unique_best)
        last = next(best_iter)

        while x := next(best_iter, None):
            a = f_beta_zero_point(
                recall[last, n], precision[last, n], recall[x, n], precision[x, n]
            )
            ax.vlines(a, 0.0, 1.0, color="k", lw=1, ls=":")
            ax.text(
                a,
                0.92,
                f"{a:.2f}",
                rotation=90,
                fontsize="x-small",
                horizontalalignment="right",
            )

            last = x

    plt.tight_layout()
    plt.savefig(f"figures/02_{ds_name}.png")
    plt.savefig(f"figures/02_{ds_name}.pdf")
    plt.close()
