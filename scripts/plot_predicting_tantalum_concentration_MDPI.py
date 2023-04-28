import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pickle
import os


# fitting function
def func(X, a, b, c):
    return a * (2 ** (-b / np.sqrt(X)) + c)

# inverse fitting function
def func_inv(X, a, b, c):
    return ((-b * np.log(2)) / np.log(X / a - c)) ** 2

# Function to plot intervals
def plot_interval(U0, U1, I, a, b, c, text1, text2, color, start_Z, start_U, mode="both"):
    Z0 = func_inv(U0 / I, a, b, c)
    Z1 = func_inv(U1 / I, a, b, c)
    print(Z0, Z1)
    if mode in ["horizontal", "both"]:
        ax.fill_between([start_Z, Z0, Z1], [U0, U0, U1], [U1, U1, U1], color=color, alpha=0.15)
    if mode in ["vertical", "both"]:
        ax.fill_between([Z0, Z0], [start_U, start_U], [U1, U1], color=color, alpha=0.15)
    ax.annotate("", xy=(Z0, U0), xytext=(start_Z, U0),
                arrowprops=dict(arrowstyle="->", color=color, linewidth=1))
    ax.annotate("", xy=(Z1, U1), xytext=(start_Z, U1),
                arrowprops=dict(arrowstyle="->", color=color, linewidth=1))
    start = start_Z+0.5
    ax.text(start, U1 + 0.05, text2, color=color, verticalalignment="center", fontsize=10)
    ax.text(start, U0 + 0.05, text1, color=color, verticalalignment="center", fontsize=10)

# Values extracted from images
corrected = [0.91255862, 1.68910593, 1.94510587, 2.08347034, 2.17454121, 2.80431447,
             2.80575022, 3.19268558, 3.83540971, 2.28765629, 1.95726733, 1.66752555]

raw = [0.91461182, 1.6973877,  1.953125,   2.09075928, 2.17468262, 2.79968262,
       2.7923584,  3.17047119, 3.81530762, 2.28637695, 1.96044922, 1.67205811]

# Make into np.arrays
corrected = np.array(corrected)
raw = np.array(raw)

Z = np.array([13, 22, 26, 28, 29, 41, 42, 50, 74, 30.5, 25.4, 21.3])

names = ["Al", "Ti", "Fe", "Ni", "Cu", "Nb", "Mo", "Sn", "W", "Bronze", "St. Steel", "Ti64"]

eps = 0.558

a, b, c = 4.987, 10.452, -0.043


params = {'text.latex.preamble': r"\usepackage{lmodern}",
          'text.usetex': True,
          'font.size': 10,
          'font.family': 'lmodern'}
plt.rcParams.update(params)
fig, ax = plt.subplots()
fig.set_size_inches(4.49, 3.49)

# plot fitting for 2 mA
start_Z = 10
start_U = func(start_Z, a, b, c) * 2.0
end_Z = 70
x = np.linspace(start_Z, end_Z)
y = func(x, a, b, c) * 2.0
ax.plot(x, y, "--", linewidth=2, color="C3")
ax.plot(x + eps, y, "--", linewidth=1, color="C3", alpha=0.5)
ax.plot(x - eps, y, "--", linewidth=1, color="C3", alpha=0.5)
ax.fill_between(x, [func(z - eps, a, b, c) * 2 for z in x], [func(z + eps, a, b, c) * 2 for z in x], color="C3",
                alpha=0.1)

# Add U(Z) text
i = 6
ax.text(x[i] - 3, y[i] + 0.05, r"$\bar{U}(Z)$", color="C3")

# Scatterplot alloys and elements
marker_filled = matplotlib.markers.MarkerStyle(marker="o", fillstyle="full")
marker_unfilled = matplotlib.markers.MarkerStyle(marker="o", fillstyle="none")
marker_square_filled = matplotlib.markers.MarkerStyle(marker="s", fillstyle="full")
marker_square_unfilled = matplotlib.markers.MarkerStyle(marker="s", fillstyle="none")
end = 7
ax.scatter(Z[:-end], corrected[:-end], marker=marker_filled, color="C3")
ax.scatter(Z[:-end], raw[:-end], marker=marker_unfilled, color="C3")
ax.scatter(Z[-3:], corrected[-3:], marker=marker_square_filled, color="C3")
ax.scatter(Z[-3:], raw[-3:], marker=marker_square_unfilled, color="C3")

# Add text
for x0, y0, n in zip(Z[:-end], corrected[:-end], names[:-end]):
    ax.text(x0 + 0.6, y0 - 0.06, n, color="C3", verticalalignment="center", fontsize=10)

for x0, y0, n in zip(Z[-3:], corrected[-3:], names[-3:]):
    ax.text(x0 - 0.6, y0 + 0.09, n, color="C3", va="center", ha="right", fontsize=10)

# Add Zeff
Z_TNM = 20.55861498
U_TNM = 1.589084461
Z_TNM_13Ta = 24.7545353
U_TNM_13Ta = 1.894449744

# Vertical
arrowprops=dict(arrowstyle="->", color="C0", linewidth=1)
ax.annotate("", xy=(Z_TNM, U_TNM), xytext=(Z_TNM, 0.79), arrowprops=arrowprops)
ax.annotate("", xy=(Z_TNM_13Ta, U_TNM_13Ta), xytext=(Z_TNM_13Ta, 0.79), arrowprops=arrowprops)
xx = np.linspace(Z_TNM, Z_TNM_13Ta)
ax.fill_between(xx, func(xx, a, b, c) * 2, xx*0, color="C0", alpha=0.15)
ax.text(Z_TNM + 0.25, 0.825, r'$\bar{Z}^{\mathrm{eff}}_{\mathrm{TNM}}$', ha="left", va="bottom", color="C0")
ax.text(Z_TNM_13Ta + 0.25, 0.825, r'$\bar{Z}^{\mathrm{eff}}_{\mathrm{13Ta}}$', ha="left", va="bottom", color="C0")

# Horizontal
plot_interval(1.7, 1.9, 2.0, a, b, c, r"$U_{\mathrm{TNM}}$", r"$U_{\mathrm{INCLUSION}}$", "C2", start_Z=12,
              start_U=0.8, mode="horizontal")

ax.axis([12, 32, 0.8, 2.5])
ax.set_xlabel("Effective Atomic Number $Z^\mathrm{eff}$")
ax.set_ylabel("Detector Voltage [V]")
ax.set_xticks(np.arange(13, 32), np.arange(13, 32))

cwd = os.getcwd()
plot_path = os.path.join(cwd, "..", "figures", "predicting_tantalum_concentration_MDPI.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")

quit()

Z_TNM = 20.571
U_TNM = 0.728 * 2.0

Z_TNM_13Ta = 24.691
U_TNM_13Ta = 0.866 * 2.0

Z_TNM_27Ta = 29.492
U_TNM_27Ta = 1.006 * 2.0

Z_Ta = 73
U_Ta = func(Z_Ta, a, b, c) * 2.0

a_Ta = np.arange(0, 0.25, 0.01)
U_Ta_TNM = U_TNM * (1 - a_Ta) + U_Ta * a_Ta

y2 = (y - U_TNM) / (U_Ta - U_TNM)

points_y = np.array([1.7, 1.75, 1.85, 1.9])
points_x = func_inv(points_y / 2.0, a, b, c)

x0 = [Z_TNM, Z_TNM_13Ta, Z_TNM_27Ta]
y_end = [func(z, a, b, c) * 2 for z in x0]
# ax.vlines(x0, y_start, y_end, linestyles="-", color="C0")
for i in range(2):
    ax.annotate("", xy=(x0[i], y_end[i]), xytext=(x0[i], func(18.95, a, b, c) * 2),
                arrowprops=dict(arrowstyle="->", color="C0", linewidth=1))
# ax.hlines(y_end, y_start, x0, linestyles=":", color="C0")
ax.fill_between([Z_TNM, Z_TNM_13Ta], [y_end[0], y_end[1]], [0, 0], color="C0", alpha=0.15)
# ax.text(Z_TNM_13Ta+0.25, 1.35, 'Expected $Z^*$ Span', ha="left", va="bottom", rotation=-90, color="C0")
ax.text(Z_TNM + 0.25, 1.475, r'$\bar{Z}^{\mathrm{eff}}_{\mathrm{TNM}}$', ha="left", va="bottom", color="C0")
ax.text(Z_TNM_13Ta + 0.25, 1.475, r'$\bar{Z}^{\mathrm{eff}}_{\mathrm{13Ta}}$', ha="left", va="bottom", color="C0")
# ax.text(Z_TNM_27Ta-0.25, 1.35, r'$\bar{Z}^{\mathrm{eff}}_{\mathrm{27Ta}}$', ha="right", va="bottom", color="C0")
# ax.annotate("", xy=(24.17, y_end[0]), xytext=(Z_TNM, y_end[0]), arrowprops=dict(arrowstyle="<->", color="k", linestyle=":"))
xy = ((24.17 + Z_TNM) / 2, y_end[0] - 0.03)
# ax.annotate(r"$3.6\, Z^{\mathrm{eff}}$", xy=xy, xytext=xy, ha="center", va="center", color="k")
# ax.plot([23.9, 23.9], [y_end[0]-0.025, y_end[0]+0.025], color="k", linestyle="-", linewidth=1)
# ax.annotate("", xy=(27.5, 1.6), xytext=(Z_TNM_13Ta, 1.6), arrowprops=dict(arrowstyle="<->", color="k", linestyle=":"))
xy = ((27.5 + Z_TNM_13Ta) / 2, 1.6 - 0.03)
# ax.annotate("$2.8\, Z^{\mathrm{eff}}$", xy=xy, xytext=xy, ha="center", va="center", color="k")
# ax.plot([27.2, 27.2], [1.6-0.025, 1.6+0.025], color="k", linestyle="-", linewidth=1)

# ax.text(start_Z + 1.2, y_end[0]+0.025, "0\% Ta", color="C0", verticalalignment="center", fontsize=10)
# ax.text(start_Z + 1.2, y_end[1]-0.075, "13\% Ta", color="C0", verticalalignment="center", fontsize=10)
# ax.text(start_Z + 1.2, y_end[2]+0.025, "27\% Ta", color="C0", verticalalignment="center", fontsize=10)

# ax.annotate("", xy=(22.5, y_end[1] - 0.01), xytext=(21.25, y_end[1]-0.075), arrowprops=dict(arrowstyle="->", color="C0"))

marker_filled = matplotlib.markers.MarkerStyle(marker="o", fillstyle="full")
marker_unfilled = matplotlib.markers.MarkerStyle(marker="o", fillstyle="none")
marker_square_filled = matplotlib.markers.MarkerStyle(marker="s", fillstyle="full")
marker_square_unfilled = matplotlib.markers.MarkerStyle(marker="s", fillstyle="none")
ax.scatter([22, 26, 28, 29], np.array([1.55, 1.78, 1.91, 1.99]) * f, marker=marker_filled, color="C3")
ax.scatter([22, 26, 28, 29], np.array([1.55, 1.74, 1.84, 1.92]) * f, marker=marker_unfilled, color="C3")
ax.scatter([21.336], [1.53 * f], marker=marker_square_filled, color="C3")
ax.scatter([25.433], [0.902 * 2 * f], marker=marker_square_filled, color="C3")
ax.scatter([21.331], [1.57 * f], marker=marker_square_unfilled, color="C3")
ax.scatter([25.433], [1.85 * f], marker=marker_square_unfilled, color="C3")
# ax.text(21.5, 1.49, r"$\bar{U}(Z)$", color="C3", verticalalignment="center", fontsize=10)
for name, x, y in zip(["Ti", "Fe", "Ni", "Cu"],
                      [22, 26, 28, 29],  # [22,    26,   28,   27.5, 19.3,   22.3],
                      [1.55, 1.74, 1.84, 1.92]):  # [1.53,  1.73, 1.83, 2.05, 1.58,   1.85]):
    ax.text(x + 0.4, y * f - 0.04, name, color="C3", verticalalignment="center", fontsize=10)
for name, x, y in zip(["Ti64", "St. Steel"],
                      [19.3, 22.3],
                      [1.57, 1.85]):
    ax.text(x + 0.25, y * f - 0.025, name, color="C3", verticalalignment="center", fontsize=10)
plot_interval(1.7, 1.9, 2.0, a, b, c, r"$U_{\mathrm{TNM}}$", r"$U_{\mathrm{INCLUSION}}$", "C2", start_Z=start_Z,
              start_U=start_U, mode="horizontal")
# plot_interval(1.85, 1.9, 2.0, a, b, c, "Peak Tantalum", "C0", start_Z=start_Z, start_U=start_U)

marker = matplotlib.markers.MarkerStyle(marker="s", fillstyle="full")
ax.scatter([Z_TNM, Z_TNM_13Ta], [1.75, 1.9], marker=marker, color="C0", zorder=10)
# marker = matplotlib.markers.MarkerStyle(marker="s", fillstyle="right")
# cax.scatter([Z_TNM, Z_TNM_13Ta], [1.7, 1.9], marker=marker, color="C0")

markers = []
markers.append(matplotlib.lines.Line2D([], [], color='k', linestyle='--', label='Fitted curves'))
# ax.legend(handles=markers, fontsize=8)


ax.set_xlabel("Effective Atomic Number $Z^\mathrm{eff}$")
ax.set_ylabel("Detector Voltage [V]")
# ax2.set_xlabel("Tantalum mass concentration $c_{Ta}$ [wt\%]")

ax.axis([18.5, 31, func(19, a, b, c) * 2, func(31, a, b, c) * 2])
ax.set_xticks(np.arange(19, 31), np.arange(19, 31))

# ax2.set_xlim(ax.get_xlim())
# c_Tas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
# ax2_Us = c_Tas*2.014 + 1.456
# ax2_Zs = func_inv(ax2_Us/2, a, b, c)
# ax2.set_xticks(ax2_Zs)
# x2.set_xticklabels([int(x) for x in (ax2_Us - U_TNM)/(U_Ta - U_TNM)*100])

# plot_path = os.path.join(cwd, "..", "figures", "predicting_tantalum_concentration.pdf")
# plt.savefig(plot_path, dpi=1000, bbox_inches="tight")
# plot_path = os.path.join(cwd, "..", "..", "..", "Latex Documents", "ELO Image Analysis", "Images", "predicting_tantalum_concentration.pdf")
# plt.savefig(plot_path, dpi=1000, bbox_inches="tight")

plt.show()
