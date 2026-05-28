import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import matplotlib as mpl
import matplotlib.patheffects as pe
mpl.rcParams["hatch.linewidth"] = 8

table2 = {
    "g05_60.0": {
        "GD-IM":      {"Poly": 189.6,  "Sig": 31.5,   "Per": 44.0,   "Clip": 49.8},
        "MOM-IM":     {"Poly": 197.9,  "Sig": 31.8,   "Per": 49.1,   "Clip": 56.8},
        "1-ADAM-IM":  {"Poly": 6.0,    "Sig": 4.0,    "Per": 3.1,    "Clip": 6.9},
    },
    "g05_60.1": {
        "GD-IM":      {"Poly": 30.9,   "Sig": 48.4,   "Per": 150.0,  "Clip": 44.3},
        "MOM-IM":     {"Poly": 30.4,   "Sig": 50.8,   "Per": 134.6,  "Clip": 40.0},
        "1-ADAM-IM":  {"Poly": 1.6,    "Sig": 1.8,    "Per": 2.0,    "Clip": 3.1},
    },
    "g05_60.2": {
        "GD-IM":      {"Poly": 81.8,   "Sig": 115.7,  "Per": 237.1,  "Clip": 103.7},
        "MOM-IM":     {"Poly": 78.1,   "Sig": 114.6,  "Per": 249.5,  "Clip": 107.2},
        "1-ADAM-IM":  {"Poly": 5.6,    "Sig": 4.8,    "Per": 6.6,    "Clip": 7.2},
    },
    "g05_60.3": {
        "GD-IM":      {"Poly": 754.3,  "Sig": 83.8,   "Per": 51.8,   "Clip": 135.3},
        "MOM-IM":     {"Poly": 607.1,  "Sig": 89.3,   "Per": 53.6,   "Clip": 138.3},
        "1-ADAM-IM":  {"Poly": 91.9,   "Sig": 6.4,    "Per": 10.1,   "Clip": 14.5},
    },
    "g05_60.4": {
        "GD-IM":      {"Poly": 102.9,  "Sig": 76.6,   "Per": 97.2,   "Clip": 125.0},
        "MOM-IM":     {"Poly": 100.7,  "Sig": 68.1,   "Per": 90.0,   "Clip": 128.1},
        "1-ADAM-IM":  {"Poly": 5.3,    "Sig": 2.8,    "Per": 3.9,    "Clip": 5.6},
    },
    "g05_60.5": {
        "GD-IM":      {"Poly": 160.5,  "Sig": 124.0,  "Per": 219.4,  "Clip": 134.7},
        "MOM-IM":     {"Poly": 160.4,  "Sig": 128.0,  "Per": 237.8,  "Clip": 154.4},
        "1-ADAM-IM":  {"Poly": 11.9,   "Sig": 8.5,    "Per": 11.8,   "Clip": 8.9},
    },
    "g05_60.6": {
        "GD-IM":      {"Poly": 1398.2, "Sig": 496.8,  "Per": 938.3,  "Clip": 1137.5},
        "MOM-IM":     {"Poly": 1766.2, "Sig": 482.2,  "Per": 1011.9, "Clip": 1786.9},
        "1-ADAM-IM":  {"Poly": 48.9,   "Sig": 115.6,  "Per": 92.0,   "Clip": 55.2},
    },
    "g05_60.7": {
        "GD-IM":      {"Poly": 254.9,  "Sig": 61.1,   "Per": 97.2,   "Clip": 118.7},
        "MOM-IM":     {"Poly": 310.1,  "Sig": 62.4,   "Per": 96.6,   "Clip": 105.6},
        "1-ADAM-IM":  {"Poly": 12.2,   "Sig": 3.4,    "Per": 3.0,    "Clip": 8.7},
    },
    "g05_60.8": {
        "GD-IM":      {"Poly": 1525.8, "Sig": 239.0,  "Per": 1001.2, "Clip": 332.7},
        "MOM-IM":     {"Poly": 1365.6, "Sig": 227.6,  "Per": 1290.8, "Clip": 415.1},
        "1-ADAM-IM":  {"Poly": 21.9,   "Sig": 10.8,   "Per": 30.1,   "Clip": 10.6},
    },
    "g05_60.9": {
        "GD-IM":      {"Poly": 291.9,  "Sig": 81.8,   "Per": 267.9,  "Clip": 173.2},
        "MOM-IM":     {"Poly": 283.1,  "Sig": 101.6,  "Per": 258.3,  "Clip": 161.7},
        "1-ADAM-IM":  {"Poly": 17.1,   "Sig": 11.1,   "Per": 11.6,   "Clip": 8.4},
    },

    "g05_80.0": {
        "GD-IM":      {"Poly": 496.0,  "Sig": 254.5,  "Per": 197.6,  "Clip": 246.7},
        "MOM-IM":     {"Poly": 441.3,  "Sig": 212.2,  "Per": 187.5,  "Clip": 265.7},
        "1-ADAM-IM":  {"Poly": 23.4,   "Sig": 15.0,   "Per": 12.0,   "Clip": 28.2},
    },
    "g05_80.1": {
        "GD-IM":      {"Poly": 5.3,    "Sig": 12.9,   "Per": 23.5,   "Clip": 19.2},
        "MOM-IM":     {"Poly": 5.0,    "Sig": 13.6,   "Per": 24.8,   "Clip": 16.0},
        "1-ADAM-IM":  {"Poly": 0.5,    "Sig": 0.3,    "Per": 0.4,    "Clip": 0.3},
    },
    "g05_80.2": {
        "GD-IM":      {"Poly": 2042.1, "Sig": 158.7,  "Per": 211.9,  "Clip": 414.7},
        "MOM-IM":     {"Poly": 9990.0, "Sig": 151.1,  "Per": 217.8,  "Clip": 358.9},
        "1-ADAM-IM":  {"Poly": 94.5,   "Sig": 18.3,   "Per": 20.6,   "Clip": 67.6},
    },
    "g05_80.3": {
        "GD-IM":      {"Poly": 553.7,  "Sig": 667.7,  "Per": 1821.4, "Clip": 687.4},
        "MOM-IM":     {"Poly": 440.1,  "Sig": 533.9,  "Per": 1497.5, "Clip": 515.8},
        "1-ADAM-IM":  {"Poly": 79.7,   "Sig": 67.9,   "Per": 85.6,   "Clip": 46.9},
    },
    "g05_80.4": {
        "GD-IM":      {"Poly": 320.5,  "Sig": 119.2,  "Per": 255.3,  "Clip": 221.8},
        "MOM-IM":     {"Poly": 375.8,  "Sig": 125.3,  "Per": 296.2,  "Clip": 215.4},
        "1-ADAM-IM":  {"Poly": 26.7,   "Sig": 10.4,   "Per": 13.0,   "Clip": 55.0},
    },
    "g05_80.5": {
        "GD-IM":      {"Poly": 3871.9, "Sig": 893.2,  "Per": 1011.9, "Clip": 1735.2},
        "MOM-IM":     {"Poly": 2850.6, "Sig": 917.4,  "Per": 2859.4, "Clip": 1155.8},
        "1-ADAM-IM":  {"Poly": 91.9,   "Sig": 104.8,  "Per": 82.7,   "Clip": 75.6},
    },
    "g05_80.6": {
        "GD-IM":      {"Poly": 426.5,  "Sig": 124.6,  "Per": 269.5,  "Clip": 165.6},
        "MOM-IM":     {"Poly": 390.6,  "Sig": 117.0,  "Per": 244.9,  "Clip": 175.1},
        "1-ADAM-IM":  {"Poly": 102.7,  "Sig": 29.9,   "Per": 21.3,   "Clip": 122.6},
    },
    "g05_80.7": {
        "GD-IM":      {"Poly": 2159.0, "Sig": 208.2,  "Per": 166.2,  "Clip": 558.6},
        "MOM-IM":     {"Poly": 1803.0, "Sig": 211.6,  "Per": 149.7,  "Clip": 498.6},
        "1-ADAM-IM":  {"Poly": 64.4,   "Sig": 19.7,   "Per": 5.8,    "Clip": 239.2},
    },
    "g05_80.8": {
        "GD-IM":      {"Poly": 2269.5, "Sig": 1308.5, "Per": 1287.8, "Clip": 1321.9},
        "MOM-IM":     {"Poly": 2655.1, "Sig": 1139.1, "Per": 1582.2, "Clip": 1316.2},
        "1-ADAM-IM":  {"Poly": 189.3,  "Sig": 48.3,   "Per": 98.9,   "Clip": 239.2},
    },
    "g05_80.9": {
        "GD-IM":      {"Poly": 292.8,  "Sig": 262.8,  "Per": 695.6,  "Clip": 355.5},
        "MOM-IM":     {"Poly": 313.2,  "Sig": 274.0,  "Per": 688.5,  "Clip": 487.6},
        "1-ADAM-IM":  {"Poly": 37.8,   "Sig": 20.7,   "Per": 25.2,   "Clip": 36.7},
    },

    "g05_100.0": {
        "GD-IM":      {"Poly": 168.5,  "Sig": 317.0,  "Per": 273.2,  "Clip": 586.5},
        "MOM-IM":     {"Poly": 172.3,  "Sig": 318.8,  "Per": 236.8,  "Clip": 568.9},
        "1-ADAM-IM":  {"Poly": 14.3,   "Sig": 14.0,   "Per": 14.0,   "Clip": 128.8},
    },
    "g05_100.1": {
        "GD-IM":      {"Poly": 230.4,  "Sig": 298.3,  "Per": 1225.8, "Clip": 344.3},
        "MOM-IM":     {"Poly": 208.8,  "Sig": 308.0,  "Per": 1185.6, "Clip": 372.4},
        "1-ADAM-IM":  {"Poly": 14.6,   "Sig": 18.3,   "Per": 17.1,   "Clip": 49.1},
    },
    "g05_100.2": {
        "GD-IM":      {"Poly": 93.5,   "Sig": 215.4,  "Per": 550.8,  "Clip": 369.5},
        "MOM-IM":     {"Poly": 94.0,   "Sig": 211.8,  "Per": 581.1,  "Clip": 411.3},
        "1-ADAM-IM":  {"Poly": 3.3,    "Sig": 5.1,    "Per": 5.2,    "Clip": 202.4},
    },
    "g05_100.3": {
        "GD-IM":      {"Poly": 1088.5, "Sig": 1219.2, "Per": 3938.7, "Clip": 2195.8},
        "MOM-IM":     {"Poly": 644.9,  "Sig": 1499.9, "Per": 3341.4, "Clip": 2028.9},
        "1-ADAM-IM":  {"Poly": 402.2,  "Sig": 293.3,  "Per": 224.3,  "Clip": 312.8},
    },
    "g05_100.4": {
        "GD-IM":      {"Poly": 7612.2, "Sig": 227.7,  "Per": 162.2,  "Clip": 422.0},
        "MOM-IM":     {"Poly": 7877.0, "Sig": 243.8,  "Per": 170.9,  "Clip": 442.8},
        "1-ADAM-IM":  {"Poly": 55.2,   "Sig": 22.4,   "Per": 17.3,   "Clip": 73.6},
    },
    "g05_100.5": {
        "GD-IM":      {"Poly": 1427.9, "Sig": 792.1,  "Per": 213.2,  "Clip": 1630.7},
        "MOM-IM":     {"Poly": 1591.0, "Sig": 716.2,  "Per": 224.0,  "Clip": 2388.7},
        "1-ADAM-IM":  {"Poly": 67.3,   "Sig": 80.4,   "Per": 49.3,   "Clip": 81.6},
    },
    "g05_100.6": {
        "GD-IM":      {"Poly": 443.0,  "Sig": 285.7,  "Per": 530.6,  "Clip": 382.4},
        "MOM-IM":     {"Poly": 488.5,  "Sig": 288.3,  "Per": 520.5,  "Clip": 248.0},
        "1-ADAM-IM":  {"Poly": 80.6,   "Sig": 15.6,   "Per": 18.7,   "Clip": 46.4},
    },
    "g05_100.7": {
        "GD-IM":      {"Poly": 395.0,  "Sig": 397.6,  "Per": 292.8,  "Clip": 774.4},
        "MOM-IM":     {"Poly": 390.9,  "Sig": 338.7,  "Per": 347.5,  "Clip": 662.9},
        "1-ADAM-IM":  {"Poly": 112.1,  "Sig": 30.6,   "Per": 14.0,   "Clip": 124.0},
    },
    "g05_100.8": {
        "GD-IM":      {"Poly": 1750.2, "Sig": 1091.2, "Per": 2745.3, "Clip": 1258.1},
        "MOM-IM":     {"Poly": 1798.5, "Sig": 1110.7, "Per": 2751.1, "Clip": 1586.6},
        "1-ADAM-IM":  {"Poly": 368.0,  "Sig": 258.5,  "Per": 373.0,  "Clip": 331.2},
    },
    "g05_100.9": {
        "GD-IM":      {"Poly": 189.7,  "Sig": 249.1,  "Per": 1018.9, "Clip": 356.6},
        "MOM-IM":     {"Poly": 160.0,  "Sig": 238.3,  "Per": 1148.9, "Clip": 435.3},
        "1-ADAM-IM":  {"Poly": 23.9,   "Sig": 25.7,   "Per": 24.2,   "Clip": 252.7},
    },
}

# Print in the same order as the table for manual checking
for inst, vals in table2.items():
    gd = vals["GD-IM"]
    mom = vals["MOM-IM"]
    adam1 = vals["1-ADAM-IM"]
    print(
        f"{inst:10s} | "
        f"GD-IM:      Poly={gd['Poly']:8.1f}  Sig={gd['Sig']:8.1f}  Per={gd['Per']:8.1f}  Clip={gd['Clip']:8.1f} | "
        f"MOM-IM:     Poly={mom['Poly']:8.1f}  Sig={mom['Sig']:8.1f}  Per={mom['Per']:8.1f}  Clip={mom['Clip']:8.1f} | "
        f"1-ADAM-IM:  Poly={adam1['Poly']:8.1f}  Sig={adam1['Sig']:8.1f}  Per={adam1['Per']:8.1f}  Clip={adam1['Clip']:8.1f}"
    )


def plot_table2_normalized_averages(
    pdf_path=r"/Users/stijnvanvooren/Documents/PhD/Adam/Results/Adam_Final/plots/Table2_g05_normalized_TTS.pdf"
):
    # -------------------------------------------------
    # Methods in plotting order
    # -------------------------------------------------
    methods = [
        ("GD-IM", "Poly"),
        ("GD-IM", "Sig"),
        ("GD-IM", "Per"),
        ("GD-IM", "Clip"),
        ("MOM-IM", "Poly"),
        ("MOM-IM", "Sig"),
        ("MOM-IM", "Per"),
        ("MOM-IM", "Clip"),
        ("1-ADAM-IM", "Poly"),
        ("1-ADAM-IM", "Sig"),
        ("1-ADAM-IM", "Per"),
        ("1-ADAM-IM", "Clip"),
    ]

    # Legend labels
    method_labels = [
        "GD-IM Poly", "GD-IM Sigmoid", "GD-IM Periodic", "GD-IM Clipped",
        "MOM-IM Poly", "MOM-IM Sigmoid", "MOM-IM Periodic", "MOM-IM Clipped",
        "1-ADAM-IM Poly", "1-ADAM-IM Sigmoid", "1-ADAM-IM Periodic", "1-ADAM-IM Clipped",
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
            vals = [table2[inst][opt][nl] for opt, nl in methods]
            min_val = min(vals)
            normalized_vals = [v / min_val for v in vals]
            per_instance_normalized.append(normalized_vals)

        per_instance_normalized = np.array(per_instance_normalized)
        avg_normalized[N] = per_instance_normalized.mean(axis=0)

    # -------------------------------------------------
    # Print values for checking
    # -------------------------------------------------
    print("\nAveraged normalized TTS values (Table 2):\n")
    for N in [60, 80, 100]:
        print(f"N = {N}")
        for label, val in zip(method_labels, avg_normalized[N]):
            print(f"  {label:16s} : {val:.6f}")
        print()

    # -------------------------------------------------
    # Style
    # -------------------------------------------------
    bg = "white"

    # Same 4 colors repeated for the 4 nonlinearities
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

    # No hatch for GD, // for MOM, x for 1-ADAM
    hatches = [
        "", "", "", "",
        "/", "/", "/", "/",
        "x", "x", "x", "x"
    ]

    fig, ax = plt.subplots(figsize=(22, 8), constrained_layout=False)
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

    ax.tick_params(axis='y', which='major', labelsize=28, width=3.2, length=13)
    ax.tick_params(
        axis='y',
        which='minor',
        width=2.1,
        length=7
    )
    ax.tick_params(axis='x', which='major', labelsize=30, width=3.2, length=13)

    # -------------------------------------------------
    # Plot using same "current_x" logic as before
    # -------------------------------------------------
    num_methods = len(methods)   # 12
    bar_width = 0.095
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
            hatch = hatches[i]

            plot_width = bar_width * 0.75   # smaller = more space between bar edges

            # Base bar: colored fill + same-color contour
            ax.bar(
                x,
                val,
                width=plot_width,
                color=colors[i],
                edgecolor=colors[i],
                linewidth=6,
                zorder=3
            )

            # Hatch overlay: white hatch
            if hatch:
                ax.bar(
                    x,
                    val,
                    width=plot_width,
                    color="none",
                    edgecolor="white",
                    linewidth=0.0,
                    hatch=hatch,
                    zorder=4
                )

        x_positions.append(current_x + bar_width * ((num_methods - 1) / 2))
        x_labels.append(str(N))

        current_x += num_methods * bar_width + group_spacing

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=30)

    # -------------------------------------------------
    # Y axis (log scale)
    # -------------------------------------------------
    ax.set_yscale("log")

    ymin = min(v for v in row_values if v > 0)
    ymax = max(row_values)

    ymin = 10 ** np.floor(np.log10(ymin))
    ymax = 1.2*10 ** np.ceil(np.log10(ymax)-1)

    ax.set_ylim(ymin, ymax)

    ticks = [10**k for k in range(int(np.log10(ymin)), int(np.log10(ymax)) + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([rf"$10^{{{int(np.log10(t))}}}$" for t in ticks], fontsize=28)

    ax.set_ylabel(r"$\mathrm{Normalized\ TTT}$", fontsize=36, labelpad=18)
    ax.set_xlabel(r"$N$", fontsize=34, labelpad=10)

    # -------------------------------------------------
    # Legend
    # -------------------------------------------------
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

    # Reorder so displayed rows become:
    # row 1 = GD-IM
    # row 2 = MOM-IM
    # row 3 = 1-ADAM-IM
    order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    legend_elements = [legend_elements[i] for i in order]

    plt.subplots_adjust(top=0.83, bottom=0.15)

    legend = fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        fontsize=26,
        frameon=False,
        borderaxespad=0.1,
        handlelength=2.0,
        columnspacing=1.4,
        handletextpad=0.6
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


plot_table2_normalized_averages()