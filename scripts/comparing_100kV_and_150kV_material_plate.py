import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import skimage
import os
import scipy
from scipy.optimize import curve_fit


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def create_annulus_mask(h, w, center=None, inner_radius=None, outer_radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if inner_radius is None:  # use the smallest distance between the center and image walls
        inner_radius = 0
    if outer_radius is None: # use the smallest distance between the center and image walls
        inner_radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask_outer = dist_from_center <= outer_radius
    mask_inner = dist_from_center >= inner_radius
    mask = mask_inner * mask_outer
    return mask


def func(X, a, b, c):
    return a * (2**(-b / np.sqrt(X)) + c)


def func_inv(X, a, b, c):
    return ((-b * np.log(2))/np.log(X/a - c))**2


# Periodic system data
path_to_periodic_system_data = "Periodic Table of Elements.csv"
elements = np.array(["Al", "Ti", "Fe", "Ni", "Cu", "Nb", "Mo", "Sn", "W"])
Z = np.array([13, 22, 26, 28, 29, 41, 42, 50, 74])
df = pd.read_csv(path_to_periodic_system_data)
df = df[['AtomicNumber', 'Element', 'Symbol', 'AtomicMass', 'AtomicRadius', 'Density', 'MeltingPoint', 'SpecificHeat', 'NumberofShells']]
# df = df[df["Symbol"].isin(elements)]
# df = df.set_index(np.arange(1, 10))
atomic_mass = df["AtomicMass"].iloc[10:80]
density = df["Density"].iloc[10:80]
radius = df["AtomicRadius"].iloc[10:80]


# Define the elemental names, atomic number and density of the materials on the plate
elements = np.array(["Al", "Ti", "Fe", "Ni", "Cu", "Nb", "Mo", "Sn", "W", "Bronze", "Steel", "Ti64"])
Z_bronze = 29 * 0.92 + 50 * 0.08
Z_stainless_steel = (26 * 0.67 + 24 * 0.195 + 28 * 0.105 + 25 * 0.02 + 14 * 0.01) / 1
Z_Ti64 = 22 * 0.9 + 13 * 0.06 + 23 * 0.04
Z = np.array([13, 22, 26, 28, 29, 41, 42, 50, 74, Z_bronze, Z_stainless_steel, Z_Ti64])
order = np.argsort(Z)
eta = 2 ** (-9 / np.sqrt(Z))


cwd = os.getcwd()
path = os.path.join(cwd, "Images", "ELO Material Contrast images")

names = ["100kV", "150kV", "150kV polished"]
x0s = [708, 688, 755]
y0s = [751, 771, 750]
ds = [520, 520, 700]
Rs = [420, 420, 550]
rs = [50, 50, 70]
dphis = - np.array([1, 1, 0]) * np.pi/200
Ib_maxs = [5.1, 5.1, 5.1]

fig_ELO, axes_ELO = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
fig_viol, axes_viol = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
axes = [(axes_ELO[i], axes_viol[i]) for i in range(3)]

figsize=(12, 12)
fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=figsize)

data = dict()
for name, x0, y0, d, R, r, dphi, Ib_max, ax in zip(names, x0s, y0s, ds, Rs, rs, dphis, Ib_maxs, axes):
    path = os.path.join(cwd, "Images", "ELO Material Contrast images")
    folder = os.path.join(path, name)
    paths_to_png = [os.path.join(path, img) for img in os.listdir(folder) if img.endswith(".png")]
    timestamps = [png.split("\\")[-1].split("_")[0] for png in paths_to_png]
    beam_currents = np.arange(0.5, Ib_max, 0.5)
    paths_to_tif = [os.path.join(folder, "Metadata", "%s_ORIG.tif" % ts) for ts in timestamps]

    raw_data = dict()
    corrected_data = dict()
    for path, current in zip(paths_to_tif, beam_currents):
        raw_image = plt.imread(path)
        bit_image = (raw_image - 2 ** 15) / 2 ** 15 * 10
        image = bit_image[y0 - d:y0 + d, x0 - d:x0 + d]
        m, n = np.shape(image)

        hist_data = []
        shading_compensation = []
        for j in range(12):
            phi = (j - 3) * np.pi / 6 + dphi
            x = d + R * np.cos(phi)
            y = d + R * np.sin(phi)

            mask = create_annulus_mask(m, n, [x, y], 1.5 * r, 1.75 * r)
            shading_compensation.append(np.median(image[mask]))

            mask = create_circular_mask(m, n, [x, y], r)
            hist_data.append(image[mask])

        raw_data[current] = hist_data
        hist_data = np.array(hist_data, dtype=object) / np.array(shading_compensation) * np.mean(shading_compensation)
        corrected_data[current] = hist_data

        if current == beam_currents[-1]:
            ax[0].imshow(image, cmap="gray", vmax=10)
            for j in range(12):
                phi = (j - 3) * np.pi / 6 + dphi
                x = d + R * np.cos(phi)
                y = d + R * np.sin(phi)
                ax[0].plot(x, y, "+", color="C0")

                theta = np.linspace(0, 2 * np.pi)
                alpha = 0.7
                ax[0].plot(x + r * np.cos(theta), y + r * np.sin(theta), "-", color="C0", alpha=alpha)
                ax[0].plot(x + 1.5 * r * np.cos(theta), y + 1.5 * r * np.sin(theta), "-", color="C1", alpha=alpha)
                ax[0].plot(x + 1.75 * r * np.cos(theta), y + 1.75 * r * np.sin(theta), "-", color="C1", alpha=alpha)

    data[name] = dict()
    for key, value in corrected_data.items():
        ax[1].violinplot(value)
        median_value = [np.median(v) for v in value]
        data[name][key] = np.array(median_value)
        if name == "100kV":
            marker = "x"
            ax_scatter.scatter(Z, median_value, marker=marker)
        elif name == "150kV":
            marker = "+"
            ax_scatter.scatter(Z, median_value, marker=marker)
        else:
            marker = "o"
            color = "C%i" % (2*key - 1)
            ax_scatter.scatter(Z, median_value, marker=marker, facecolors="none", edgecolors=color)
            popt, pcov = curve_fit(func, Z[:-3], median_value[:-3]/key)
            a, b, c = popt
            ax_scatter.plot(np.linspace(10, 80), func(np.linspace(10, 80), a, b, c) * key, "--")
            ax_scatter.text(80.1, func(81, *popt)*key, "%.1f mA" % key, color=color, verticalalignment="center", fontsize=16)

    ax[1].set_xticks(ticks=np.arange(1, 13), labels=elements)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

df = pd.DataFrame(data)
difference = df["150kV"] / df["100kV"]
fig_diff, ax_diff = plt.subplots(1, 1, figsize=figsize)
cmap = matplotlib.cm.get_cmap('viridis')
for i, element in enumerate(elements):
    diff = np.array([v[i] for k, v in difference.items()])
    beam_currents = np.arange(0.5, Ib_maxs[0], 0.5)
    ax_diff.plot(beam_currents, diff, label=element, color=cmap(i/11))
    ax_diff.legend()
    ax_diff.set_xticks(beam_currents, beam_currents)
    ax_diff.set_xlabel("Beam Current [mA]")
    ax_diff.set_ylabel("Detector Voltage Ratio")
    ax_diff.set_title(" $U_{150 kV}/U_{100 kV}$")


    #plt.tight_layout()
    #plt.savefig("Violinplot_multicurrent.png")

for i, name in enumerate(["100 kV unpolished", "150 kV unpolished", "150 kV polished"]):
    axes_ELO[i].set_title(name)
    axes_viol[i].set_title(name)
    axes_viol[i].set_ylabel("Detector voltage [V]")

markers = []
for mark, name in zip(["x", "+", "o"], ["100 kV unpolished", "150 kV unpolished", "150 kV polished"]):
    marker = matplotlib.markers.MarkerStyle(marker=mark, fillstyle="none")
    marker = matplotlib.lines.Line2D([], [], color='k', linestyle='', marker=marker, label=name)
    markers.append(marker)

markers.append(matplotlib.lines.Line2D([], [], color='k', linestyle='--', label='Fitted curves'))
ax_scatter.legend(handles=markers, fontsize=16)

ax_scatter.set_ylabel("Detector Voltage [V]")
ax_scatter.set_xlabel("Atomic Number Z")
ax_scatter.axis([5, 90, -0.5, 10.5])
ax_scatter.set_title("Detector intensities vs. Atomic number")

plt.tight_layout()

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig('figure%d.png' % i)

plt.show()