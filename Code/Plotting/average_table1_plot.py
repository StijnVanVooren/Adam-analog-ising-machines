import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import matplotlib as mpl
import matplotlib.patheffects as pe
mpl.rcParams["hatch.linewidth"] = 8


table1 = {
    "g05_60.0": {
        "ADAM-IM":   {"Poly": 6.05,  "Sig": 3.33,  "Per": 1.91,  "Clip": 3.62},
        "1-ADAM-IM": {"Poly": 5.70,  "Sig": 3.12,  "Per": 2.10,  "Clip": 4.29},
    },
    "g05_60.1": {
        "ADAM-IM":   {"Poly": 1.47,  "Sig": 1.40,  "Per": 1.55,  "Clip": 1.74},
        "1-ADAM-IM": {"Poly": 1.54,  "Sig": 1.39,  "Per": 1.36,  "Clip": 1.91},
    },
    "g05_60.2": {
        "ADAM-IM":   {"Poly": 5.35,  "Sig": 3.56,  "Per": 4.17,  "Clip": 5.60},
        "1-ADAM-IM": {"Poly": 5.29,  "Sig": 3.65,  "Per": 4.43,  "Clip": 4.94},
    },
    "g05_60.3": {
        "ADAM-IM":   {"Poly": 86.11, "Sig": 4.88,  "Per": 6.82,  "Clip": 12.10},
        "1-ADAM-IM": {"Poly": 95.99, "Sig": 4.96,  "Per": 6.95,  "Clip": 9.22},
    },
    "g05_60.4": {
        "ADAM-IM":   {"Poly": 5.48,  "Sig": 2.24,  "Per": 2.85,  "Clip": 3.87},
        "1-ADAM-IM": {"Poly": 5.20,  "Sig": 2.22,  "Per": 2.70,  "Clip": 3.55},
    },
    "g05_60.5": {
        "ADAM-IM":   {"Poly": 11.47, "Sig": 7.15,  "Per": 7.86,  "Clip": 6.07},
        "1-ADAM-IM": {"Poly": 11.63, "Sig": 6.72,  "Per": 8.09,  "Clip": 6.29},
    },
    "g05_60.6": {
        "ADAM-IM":   {"Poly": 84.02, "Sig": 74.81, "Per": 40.05, "Clip": 42.04},
        "1-ADAM-IM": {"Poly": 52.28, "Sig": 88.76, "Per": 80.25, "Clip": 46.25},
    },
    "g05_60.7": {
        "ADAM-IM":   {"Poly": 13.30, "Sig": 2.67,  "Per": 1.93,  "Clip": 4.91},
        "1-ADAM-IM": {"Poly": 12.09, "Sig": 2.68,  "Per": 2.06,  "Clip": 5.37},
    },
    "g05_60.8": {
        "ADAM-IM":   {"Poly": 24.27, "Sig": 8.92,  "Per": 21.66, "Clip": 4.75},
        "1-ADAM-IM": {"Poly": 21.52, "Sig": 8.48,  "Per": 20.30, "Clip": 6.72},
    },
    "g05_60.9": {
        "ADAM-IM":   {"Poly": 20.37, "Sig": 8.63,  "Per": 7.29,  "Clip": 6.79},
        "1-ADAM-IM": {"Poly": 17.03, "Sig": 8.59,  "Per": 7.98,  "Clip": 5.49},
    },

    "g05_80.0": {
        "ADAM-IM":   {"Poly": 42.49, "Sig": 19.24, "Per": 14.82, "Clip": 15.95},
        "1-ADAM-IM": {"Poly": 35.00, "Sig": 17.96, "Per": 12.88, "Clip": 28.38},
    },
    "g05_80.1": {
        "ADAM-IM":   {"Poly": 0.57,  "Sig": 0.36,  "Per": 0.44,  "Clip": 0.39},
        "1-ADAM-IM": {"Poly": 0.73,  "Sig": 0.33,  "Per": 0.46,  "Clip": 0.27},
    },
    "g05_80.2": {
        "ADAM-IM":   {"Poly": 136.42, "Sig": 20.45, "Per": 20.62, "Clip": 81.04},
        "1-ADAM-IM": {"Poly": 137.61, "Sig": 22.09, "Per": 22.16, "Clip": 69.40},
    },
    "g05_80.3": {
        "ADAM-IM":   {"Poly": 115.39, "Sig": 87.91, "Per": 62.76, "Clip": 34.16},
        "1-ADAM-IM": {"Poly": 118.11, "Sig": 81.93, "Per": 100.22, "Clip": 48.59},
    },
    "g05_80.4": {
        "ADAM-IM":   {"Poly": 40.94, "Sig": 12.49, "Per": 14.05, "Clip": 40.08},
        "1-ADAM-IM": {"Poly": 39.53, "Sig": 12.54, "Per": 13.98, "Clip": 57.00},
    },
    "g05_80.5": {
        "ADAM-IM":   {"Poly": 135.32, "Sig": 125.37, "Per": 75.98, "Clip": 59.61},
        "1-ADAM-IM": {"Poly": 142.83, "Sig": 126.82, "Per": 101.48, "Clip": 76.63},
    },
    "g05_80.6": {
        "ADAM-IM":   {"Poly": 123.98, "Sig": 36.73, "Per": 25.93, "Clip": 237.80},
        "1-ADAM-IM": {"Poly": 149.08, "Sig": 36.10, "Per": 22.92, "Clip": 120.19},
    },
    "g05_80.7": {
        "ADAM-IM":   {"Poly": 111.81, "Sig": 23.28, "Per": 5.45, "Clip": 154.75},
        "1-ADAM-IM": {"Poly": 92.64,  "Sig": 23.59, "Per": 6.17, "Clip": 255.62},
    },
    "g05_80.8": {
        "ADAM-IM":   {"Poly": 265.74, "Sig": 70.94, "Per": 118.85, "Clip": 180.58},
        "1-ADAM-IM": {"Poly": 283.59, "Sig": 57.69, "Per": 105.90, "Clip": 245.70},
    },
    "g05_80.9": {
        "ADAM-IM":   {"Poly": 63.56, "Sig": 23.50, "Per": 30.86, "Clip": 38.73},
        "1-ADAM-IM": {"Poly": 54.58, "Sig": 24.71, "Per": 26.90, "Clip": 36.57},
    },

    "g05_100.0": {
        "ADAM-IM":   {"Poly": 26.99, "Sig": 22.72, "Per": 22.61, "Clip": 181.64},
        "1-ADAM-IM": {"Poly": 28.22, "Sig": 23.39, "Per": 21.36, "Clip": 186.43},
    },
    "g05_100.1": {
        "ADAM-IM":   {"Poly": 30.31, "Sig": 33.45, "Per": 25.92, "Clip": 128.37},
        "1-ADAM-IM": {"Poly": 28.95, "Sig": 30.85, "Per": 26.05, "Clip": 71.44},
    },
    "g05_100.2": {
        "ADAM-IM":   {"Poly": 6.74, "Sig": 8.80, "Per": 8.28, "Clip": 243.13},
        "1-ADAM-IM": {"Poly": 6.50, "Sig": 8.46, "Per": 8.01, "Clip": 288.43},
    },
    "g05_100.3": {
        "ADAM-IM":   {"Poly": 535.04, "Sig": 450.77, "Per": 416.69, "Clip": 400.19},
        "1-ADAM-IM": {"Poly": 784.09, "Sig": 494.97, "Per": 346.06, "Clip": 438.42},
    },
    "g05_100.4": {
        "ADAM-IM":   {"Poly": 130.59, "Sig": 37.17, "Per": 26.02, "Clip": 124.61},
        "1-ADAM-IM": {"Poly": 119.41, "Sig": 37.30, "Per": 26.64, "Clip": 109.22},
    },
    "g05_100.5": {
        "ADAM-IM":   {"Poly": 136.72, "Sig": 145.45, "Per": 99.62, "Clip": 82.92},
        "1-ADAM-IM": {"Poly": 140.46, "Sig": 135.34, "Per": 82.96, "Clip": 116.76},
    },
    "g05_100.6": {
        "ADAM-IM":   {"Poly": 181.31, "Sig": 28.40, "Per": 30.80, "Clip": 64.56},
        "1-ADAM-IM": {"Poly": 161.03, "Sig": 26.12, "Per": 28.74, "Clip": 65.18},
    },
    "g05_100.7": {
        "ADAM-IM":   {"Poly": 257.54, "Sig": 50.39, "Per": 22.97, "Clip": 222.63},
        "1-ADAM-IM": {"Poly": 229.64, "Sig": 50.78, "Per": 21.36, "Clip": 176.88},
    },
    "g05_100.8": {
        "ADAM-IM":   {"Poly": 641.96, "Sig": 536.13, "Per": 767.98, "Clip": 549.82},
        "1-ADAM-IM": {"Poly": 749.28, "Sig": 429.45, "Per": 573.94, "Clip": 469.38},
    },
    "g05_100.9": {
        "ADAM-IM":   {"Poly": 48.78, "Sig": 44.37, "Per": 43.05, "Clip": 335.68},
        "1-ADAM-IM": {"Poly": 46.58, "Sig": 42.51, "Per": 36.94, "Clip": 351.90},
    },
}

# Print in the same order as the table for manual checking
for inst, vals in table1.items():
    a = vals["ADAM-IM"]
    b = vals["1-ADAM-IM"]
    print(
        f"{inst:9s} | "
        f"ADAM-IM:   Poly={a['Poly']:7.2f}  Sig={a['Sig']:7.2f}  Per={a['Per']:7.2f}  Clip={a['Clip']:7.2f} | "
        f"1-ADAM-IM: Poly={b['Poly']:7.2f}  Sig={b['Sig']:7.2f}  Per={b['Per']:7.2f}  Clip={b['Clip']:7.2f}"
    )



def plot_table1_normalized_averages(pdf_path=r"/Users/stijnvanvooren/Documents/PhD/Adam/Results/Adam_Final/plots/Table1_g05_normalized_TTS.pdf"):
    # -------------------------------------------------
    # Methods in plotting order
    # -------------------------------------------------
    methods = [
        ("ADAM-IM", "Poly"),
        ("ADAM-IM", "Sig"),
        ("ADAM-IM", "Per"),
        ("ADAM-IM", "Clip"),
        ("1-ADAM-IM", "Poly"),
        ("1-ADAM-IM", "Sig"),
        ("1-ADAM-IM", "Per"),
        ("1-ADAM-IM", "Clip"),
    ]

    # Short legend labels so they fit cleanly
    method_labels = [
        "ADAM-IM Poly", "ADAM-IM Sigmoid", "ADAM-IM Periodic", "ADAM-IM Clipped",
        "1-ADAM-IM Poly", "1-ADAM-IM Sigmoid", "1-ADAM-IM Periodic", "1-ADAM-IM Clipped"
    ]

    # -------------------------------------------------
    # Group instances by N
    # -------------------------------------------------
    groups = {
        60:  [f"g05_60.{i}" for i in range(10)],
        80:  [f"g05_80.{i}" for i in range(10)],
        100: [f"g05_100.{i}" for i in range(10)],
    }

    # -------------------------------------------------
    # Normalize per instance, then average over 10 instances
    # -------------------------------------------------
    avg_normalized = {N: [] for N in groups}

    for N, inst_list in groups.items():
        per_instance_normalized = []

        for inst in inst_list:
            vals = [table1[inst][opt][nl] for opt, nl in methods]
            min_val = min(vals)
            normalized_vals = [v / min_val for v in vals]
            per_instance_normalized.append(normalized_vals)

        per_instance_normalized = np.array(per_instance_normalized)
        avg_normalized[N] = per_instance_normalized.mean(axis=0)

    # -------------------------------------------------
    # Print values for checking
    # -------------------------------------------------
    print("\nAveraged normalized TTS values:\n")
    for N in [60, 80, 100]:
        print(f"N = {N}")
        for label, val in zip(method_labels, avg_normalized[N]):
            print(f"  {label:10s} : {val:.6f}")
        print()

    # -------------------------------------------------
    # Style (matched as closely as possible to your barplots)
    # -------------------------------------------------
    bg = "white"
    colors = [
        (176/255, 67/255, 87/255),
        (88/255, 132/255, 181/255),
        (86/255, 158/255, 119/255),
        (201/255, 164/255, 63/255),
        (176/255, 67/255, 87/255),
        (88/255, 132/255, 181/255),
        (86/255, 158/255, 119/255),
        (201/255, 164/255, 63/255),
    ]
    hatches = ["", "", "", "", "/", "/", "/", "/"]

    fig, ax = plt.subplots(figsize=(18, 7), constrained_layout=False)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_axisbelow(True)

    ax.yaxis.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=2,
        alpha=0.4,
        color="0.25",
        zorder=0
    )

    ax.tick_params(axis='y', which='major', labelsize=28, width=2.2, length=9)
    ax.tick_params(axis='x', which='major', labelsize=30, width=2.2, length=6)

    # -------------------------------------------------
    # Plot using same "current_x" logic as your original barplot
    # -------------------------------------------------
    num_methods = len(methods)
    bar_width = 0.12
    group_spacing = 0.22

    x_positions = []
    x_labels = []
    current_x = 0
    row_values = []

    for N in [60, 80, 100]:
        vals = avg_normalized[N]
        row_values.extend(vals)

        for i, val in enumerate(vals):
            x = current_x + i * bar_width
            # Base bar: colored fill + same-color contour
            plot_width = bar_width * 0.79
            ax.bar(
                x,
                val,
                width=plot_width,
                color=colors[i],
                edgecolor=colors[i],
                linewidth=6.0,
                zorder=3
            )

            # Hatch overlay: white hatch
            if hatches[i]:
                ax.bar(
                    x,
                    val,
                    width=plot_width,
                    color="none",
                    edgecolor="white",
                    linewidth=0.0,
                    hatch=hatches[i],
                    zorder=4
                )
        x_positions.append(current_x + bar_width * ((num_methods - 1) / 2))
        x_labels.append(str(N))

        current_x += num_methods * bar_width + group_spacing

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=30)

    # -------------------------------------------------
    # Y axis
    # -------------------------------------------------
    ymax = max(row_values) * 1.12
    ymax = max(10, np.ceil(ymax))   # at least 10, and round up to next integer
    ax.set_ylim(0, ymax)

    ticks = np.arange(1, 11, 1)     # 1,2,...,10
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=28)

    ax.set_ylabel(r"$\mathrm{Normalized\ TTT}$", fontsize=36, labelpad=18)
    ax.set_xlabel(r"$N$", fontsize=34, labelpad=10)

    # -------------------------------------------------
    # Legend
    # -------------------------------------------------
    # Build handles in the natural order first
    legend_elements = []

    for i in range(num_methods):
        if hatches[i]:
            p = Patch(
                facecolor=colors[i],
                edgecolor="white",      # makes hatch white
                linewidth=0,
                hatch=hatches[i],
                label=method_labels[i]
            )
            p.set_path_effects([
                pe.Stroke(linewidth=6, foreground=colors[i]),  # colored contour
                pe.Normal()
            ])
        else:
            p = Patch(
                facecolor=colors[i],
                edgecolor=colors[i],
                linewidth=1.2,
                label=method_labels[i]
            )

        legend_elements.append(p)

    # Reorder so that displayed rows become:
    # row 1 = ADAM-IM
    # row 2 = 1-ADAM-IM
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    legend_elements = [legend_elements[i] for i in order]

    # Leave room above axes for the legend
    plt.subplots_adjust(top=0.86, bottom=0.15)

    legend = fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.021),
        ncol=4,
        fontsize=24,
        frameon=False,
        borderaxespad=0.1,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    print("Saving to:", pdf_path)
    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.35
    )
    print(f"Saved plot to: {pdf_path}")
    plt.show()

plot_table1_normalized_averages(pdf_path=r"/Users/stijnvanvooren/Documents/PhD/Adam/Results/Adam_Final/plots/Table1_g05_normalized_TTS.pdf")