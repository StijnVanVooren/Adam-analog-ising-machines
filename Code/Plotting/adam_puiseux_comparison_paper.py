import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Paper-style aesthetics
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
rcParams["font.size"] = 13
rcParams["axes.labelsize"] = 13
rcParams["axes.linewidth"] = 1.2
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["xtick.major.size"] = 4
rcParams["ytick.major.size"] = 4
rcParams["legend.fontsize"] = 12
rcParams["figure.figsize"] = (5.4, 2.9)


# Colors
colors = [
        (176/255, 67/255, 87/255),   # Poly
        (88/255, 132/255, 181/255),  # Sig
        (86/255, 158/255, 119/255),  # Per
        (201/255, 164/255, 63/255),  # Clip

        (176/255, 67/255, 87/255),
        (88/255, 132/255, 181/255),
        (86/255, 158/255, 119/255),
        (201/255, 164/255, 63/255),

        (176/255, 67/255, 87/255),
        (88/255, 132/255, 181/255),
        (86/255, 158/255, 119/255),
        (201/255, 164/255, 63/255),
    ]

beta1 = 0.99
beta2 = 0.99

t = np.linspace(1, 300, 2000)

# Continuous-time Adam adaptive learning-rate factor
adam_prefactor = np.sqrt(1 - beta2**t) / (1 - beta1**t)

# First-order approximation
one_adam_approx = np.sqrt(-np.log(beta2)) / ((-np.log(beta1)) * np.sqrt(t))

fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=300)

ax.plot(
    t,
    adam_prefactor,
    color=colors[0],
    linewidth=3.0,
    label="Adam adaptive LR "+r"$\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$",
)

ax.plot(
    t,
    one_adam_approx,
    color=colors[1],
    linewidth=3.0,
    linestyle="--",
    label="1-st Order Puiseux expansion "+r"$\frac{\sqrt{-\ln(\beta_2)}}{-\ln(\beta_1)\sqrt{t}}$",
)

ax.set_xlabel(r"Time $t$")
ax.set_ylabel("Adaptive learning-rate factor")
ax.set_xlim(0, 300)

ax.grid(alpha=0.18)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg = ax.legend(
    frameon=False,
    loc="upper right",
    handlelength=2.6,
    borderaxespad=0.25,
    labelspacing=0.65,
)

for line in leg.get_lines():
    line.set_linewidth(3.2)

fig.tight_layout()

plt.savefig("adam_vs_first_order_adam.pdf", bbox_inches="tight")
plt.savefig("adam_vs_first_order_adam.png", dpi=600, bbox_inches="tight")
plt.show()
