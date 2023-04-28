import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
import pickle
import bz2

cwd = os.getcwd()
parse_data = False

np.random.seed(123456)

if parse_data:
    data_folder = os.path.join(cwd, "..", "data")
    file_path = os.path.join(data_folder, "TNM_0tilt.dat")

    file1 = open(file_path, 'r')
    Lines = file1.readlines()
    start_str = "-----------------------------------------------------------------"
    start_index = [i for i, line in enumerate(Lines) if start_str in line]

    backscattered_data = []
    absorbed_data = []
    n = len(start_index)-1  # 5001
    n = 2500
    for i in range(n):

        first_line = start_index[i]
        last_line = start_index[i+1] - 1

        # header = Lines[first_line + 1]

        metadata_cols = Lines[first_line + 3][:-1]
        metadata_vals = Lines[first_line + 4][:-2]
        metadata = dict(zip(metadata_cols.split("\t"), metadata_vals.split("\t")))

        #back_scattered = metadata["BackScattered"] == "yes"
        # displayed = metadata["Displayed"] == "yes"

        data_cols = Lines[first_line + 6][:-1]
        data_vals = Lines[first_line + 7:last_line]
        data = pd.DataFrame([line[:-1].split("\t") for line in data_vals], columns=data_cols.split("\t"))
        xyz = data[["X", "Y", "Z"]].astype(float) / 1000
        # y = data["Y"].astype(float) / 1000
        # z = data["Z"].astype(float) / 1000

        if metadata["BackScattered"] == "yes":
            backscattered_data.append(xyz)
        else:
            absorbed_data.append(xyz)

    path = os.path.join("..", "data", "Backscattered_CASINO_data.npy")
    with open(path, "wb") as f:
        # compressed = bz2.BZ2File(f, "w")
        pickle.dump(backscattered_data, f)

    path = os.path.join("..", "data", "Absorbed_CASINO_metadata.npy")
    with open(path, "wb") as f:
        pickle.dump(absorbed_data, f)

    quit()
else:
    path = os.path.join("..", "data", "Backscattered_CASINO_data.npy")
    with open(path, "rb") as f:
        # compressed_file = bz2.BZ2File(f, 'r')
        backscattered_data = pickle.load(f)

    path = os.path.join("..", "data", "Absorbed_CASINO_metadata.npy")
    with open(path, "rb") as f:
        absorbed_data = pickle.load(f)



params = {'text.latex.preamble': r"\usepackage{lmodern}",
          'text.usetex': True,
          'font.size': 10,
          'font.family': 'lmodern'}
plt.rcParams.update(params)

fig = plt.figure()
ax = fig.add_subplot(111)

fig.set_size_inches(3.49, 3.49*2/3)
ax.plot([-300, 300], [0, 0], "-", color="gray", linewidth=1)
T = 15
ax.plot([-300, 300], [-T, -T], ":", color="gray", linewidth=1)
ax.plot([-300, 300], [-100, -100], ":", color="gray", linewidth=1)



n_Absorbed = 75
n_BSE = 50
for i in range(max(n_BSE, n_Absorbed)):
    q = np.random.normal(0, 100)

    if i < n_Absorbed:
        xyz = absorbed_data[i]
        x = xyz["X"] + q
        y = xyz["Y"] + q
        z = xyz["Z"]

        color = "C0"
        alpha = 0.5
        linewidth = 0.5

        ax.plot([x[0], x[0]], [40, -z[0]], color="gray", alpha=alpha, linewidth=linewidth, zorder=0)
        ax.plot([y[0], y[0]], [40, -z[0]], color="gray", alpha=alpha, linewidth=linewidth, zorder=0)
        ax.plot(x, -z, color=color, alpha=alpha, linewidth=linewidth)
        ax.plot(y, -z, color=color, alpha=alpha, linewidth=linewidth)


    q = (np.random.rand() - 0.5) * 400
    if i < n_BSE:
        xyz = backscattered_data[i]
        x = xyz["X"] + q
        y = xyz["Y"] + q
        z = xyz["Z"]

        color = "C3"
        alpha = 1
        linewidth = 0.5
        dxyz = xyz.iloc[-2] - xyz.iloc[-1]

        angle_x = np.arctan(-dxyz["X"] / dxyz["Z"])
        angle_y = np.arctan(-dxyz["Y"] / dxyz["Z"])

        L = 25
        ax.arrow(x.iloc[-1], z.iloc[-1], L * np.sin(angle_x), L * np.cos(angle_x), head_width=2, color=color, alpha=alpha,
                 linewidth=linewidth*1.5)
        ax.arrow(y.iloc[-1], z.iloc[-1], L * np.sin(angle_y), L * np.cos(angle_y), head_width=2, color=color, alpha=alpha,
                 linewidth=linewidth*1.5)

        ax.plot([x[0], x[0]], [40, -z[0]], color="gray", alpha=alpha/2, linewidth=linewidth, zorder=0)
        ax.plot([y[0], y[0]], [40, -z[0]], color="gray", alpha=alpha/2, linewidth=linewidth, zorder=0)
        ax.plot(x, -z, color=color, alpha=alpha, linewidth=linewidth)
        ax.plot(y, -z, color=color, alpha=alpha, linewidth=linewidth)


reflected = matplotlib.lines.Line2D([], [], color='C3', linestyle='-', linewidth=0.5, label="Backscattered")
absorbed = matplotlib.lines.Line2D([], [], color='C0', linestyle='-', linewidth=0.5, label="Absorbed")
# surface = matplotlib.lines.Line2D([], [], color='gray', linestyle='-', linewidth=1, label="Surface (z=0µm)")
# exit_depth = matplotlib.lines.Line2D([], [], color='gray', linestyle='-.', label="Exit depth T (z=-18.3µm)")
# exit_depth = matplotlib.lines.Line2D([], [], color='gray', linestyle=':', label="Layer depth (z=-100µm)")

ax.legend(handles=[reflected, absorbed], loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=2)

ax.axis([-250, 250, -110, 40])
ax.set_yticks([0, -T, -100], [0, T, 100])
# ax.text(170, -90, "Units in $\mu{}m$", ha="center", va="center")
ax.set_xlabel("Units in µm")
# ax.set_ylabel("$\mu{}m$")


plot_path = os.path.join(cwd, "..", "..", "..", "Latex Documents", "ELO Image Analysis", "Images", "CASINO.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")

plot_path = os.path.join(cwd, "CASINOO.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")
