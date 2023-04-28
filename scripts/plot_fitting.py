import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pickle
import os


def func(X, a, b, c):
    return a * (2**(-b / np.sqrt(X)) + c)


def func_inv(X, a, b, c):
    return ((-b * np.log(2))/np.log(X/a - c))**2


cwd = os.getcwd()

file_name = "Material_Contrast_Calibration_Plate_with_large_heatshield_STSA-corrected.npy"
df_corrected = pickle.load(open(os.path.join(cwd, "..", "data", file_name), 'rb'))

file_name = "Material_Contrast_Calibration_Plate_with_large_heatshield.npy"
df_raw = pickle.load(open(os.path.join(cwd, "..", "data", file_name), 'rb'))

file_name = "Fitting_parameters_2023.npy"
df_fitting = pickle.load(open(os.path.join(cwd, "..", "data", file_name), 'rb'))

print(df_corrected.round(3))
print("")
print(df_raw.round(3))
print("")
print(df_fitting.round(4))



plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 10,
          'font.family': 'lmodern',
          'lines.linewidth': 1}
plt.rcParams.update(params)


fig, ax = plt.subplots()
fig.set_size_inches(4.49, 3.49)

Z = np.linspace(0, 80)

elements_Z = np.array([13, 22, 26, 28, 29, 41, 42, 50, 74])

currents = df_corrected.columns

marker_unfilled = matplotlib.markers.MarkerStyle(marker="o", fillstyle="none")
marker_filled = matplotlib.markers.MarkerStyle(marker="o", fillstyle="full")
for Ib in df_corrected.columns:
    a, b, c, _, _ = df_fitting[Ib]
    U = func(Z, a, b, c) * Ib
    color = "C%i" % (Ib-1)
    ax.plot(Z, U, "--", color=color)
    size = 10
    ax.scatter(elements_Z, 6*df_raw[Ib].iloc[:-3]-5*df_corrected[Ib].iloc[:-3], marker=marker_unfilled, s=size, color=color)
    ax.scatter(elements_Z, df_corrected[Ib].iloc[:-3], marker=marker_filled, s=size, color=color)
    ax.text(80.1, func(81, a, b, c)*Ib, "%.1f mA" % Ib, color=color, verticalalignment="center", weight="bold")

ax.set_xlabel("Atomic Number Z")
ax.set_ylabel("Detector Voltage [V]")
ax.axis([0, 98, 0, 11])

legend_unfilled = matplotlib.lines.Line2D([], [], color='k', linestyle='', marker=marker_unfilled, label="Uncorrected ")
legend_filled = matplotlib.lines.Line2D([], [], color='k', linestyle='', marker=marker_filled, label="STSA-corrected")
legend_fitted_line = matplotlib.lines.Line2D([], [], color='k', linestyle='--', label='Fitted curves')
ax.legend(handles=[legend_unfilled, legend_filled, legend_fitted_line])

plot_path = os.path.join(cwd, "..", "figures", "Fitting_2.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")
plot_path = os.path.join(cwd, "..", "..", "..", "Latex Documents", "ELO Image Analysis", "Images", "Fitting_2.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")

plt.show()