import numpy as np
import matplotlib.pyplot as plt
from config import *

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


recall = np.linspace(0.1, 0.9, 5, endpoint=True)
precision = np.linspace(0.1, 0.9, 5, endpoint=True)[::-1]

labels = [f"TPR:{r:.1f} PPV:{p:.1f}" for r, p in zip(recall, precision)]

fig, axs = plt.subplots(1, 1, figsize=(4 * 2.5, 4.3))

f_betas = f_beta(recall, precision, beta=betas).T
best = np.argmax(f_betas, axis=-1)
u, ind = np.unique(best, return_index=True)
unique_best = u[np.argsort(ind)]

ax = axs

ax.grid(ls=":")
ax.set_xscale("log")
ax.set_ylim(0.0, 1.0)
ax.set_xlim(np.min(betas), np.max(betas))
ax.set_ylabel("$F_\\beta$ score")
ax.set_xlabel("$\\beta$")
ax.spines[["right", "top"]].set_visible(False)

handlers = []

for b, l in zip(unique_best, labels):
    ax.plot(betas, f_betas[:, b], c="#CCCCCC")
    h = ax.plot(betas[b == best], f_betas[:, b][b == best], lw=3, label=l)
    handlers.append(h)

ax.legend(loc='lower center',ncol=5)

best_iter = iter(unique_best)
last = next(best_iter)

while x := next(best_iter, None):
    a = f_beta_zero_point(
        recall[last], precision[last], recall[x], precision[x]
    )

    ax.vlines(a, 0.0, 1.0, color="k", lw=1, ls=":")
    ax.text(
        a,
        0.90,
        f"{a:.2f}",
        rotation=90,
        fontsize="smaller",
        horizontalalignment="right",
    )

    last = x

plt.tight_layout()
plt.savefig(f"figures/sim.png")
plt.savefig(f"figures/sim.pdf")
plt.close()
