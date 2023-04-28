import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pickle
import os
import skimage
from PIL import Image


def draw_scalebar(ax, x0, xlen, y0, yheight, text, linewidth, color="w"):
    x_scalebar = np.array([x0, x0 + xlen])
    y_scalebar = np.array([y0, y0 + yheight])
    unit_array = np.array([1, 1])
    ax.plot(x_scalebar, unit_array * np.mean(y_scalebar), color=color, linewidth=linewidth)
    ax.plot(unit_array * x_scalebar[0], y_scalebar, color=color, linewidth=linewidth)
    ax.plot(unit_array * x_scalebar[1], y_scalebar, color=color, linewidth=linewidth)
    xy = (np.mean(x_scalebar), 1.45 * y_scalebar[0] - 0.5 * y_scalebar[1])
    ax.annotate(text, xy=xy, xytext=xy, horizontalalignment="center", verticalalignment="center", color=color)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cwd = os.getcwd()

params = {'text.latex.preamble': r"\usepackage{lmodern}",
          'text.usetex': True,
          'font.size': 8,
          'font.family': 'lmodern'}
matplotlib.rcParams.update(params)

image_path = os.path.join(cwd, "..", "images", "SEM_EDX_20220411", "HELIOS_TNM_ATI19_Inclusion_Test_W10_Pos_1.tif")
entire_inclusion = np.array(Image.open(image_path))

image_path = os.path.join(cwd, "..", "images", "SEM_EDX_20220411", "HELIOS_TNM_ATI19_Inclusion_Test_W10_Detail_1.tif")
crystal_structure = np.array(Image.open(image_path))

path = os.path.join(cwd, "..", "images", "SEM_EDX_20220411", "W10_Detail_1", "Ta Lα1Elementverteilungsdaten 41.csv")
tantalum = pd.read_csv(path, header=None).dropna(axis=1)
tantalum = np.array(tantalum, dtype=np.int64)
footprint = skimage.morphology.disk(13)
density = skimage.filters.rank.sum(tantalum, footprint=footprint)
density = density[:, :]
# tantalum = skimage.filters.gaussian(tantalum, sigma=1)
# tantalum = skimage.morphology.dilation(tantalum, footprint=footprint)
# tantalum = skimage.filters.gaussian(tantalum, sigma=2)
# tantalum = skimage.exposure.adjust_gamma(tantalum, gamma=5)


#fig, axes = plt.subplots(3, 1)
fig, axes = plt.subplots(1, 3)
#fig.set_size_inches(3.49, 3.49 * 1.5)
fig.set_size_inches(5.5, 3.49)
cutoff = 138
entire_inclusion = entire_inclusion[cutoff:-cutoff, :]
crystal_structure = crystal_structure[cutoff:-cutoff, :]
cmap = "gray"
axes[0].imshow(entire_inclusion, cmap=cmap)
axes[1].imshow(crystal_structure, cmap=cmap)
cmap = "gray"
axes[2].imshow(density, cmap=cmap)
cmap = plt.get_cmap("viridis")
new_cmap = truncate_colormap(cmap, 0.10, 0.90)
contour_data = skimage.filters.gaussian(density, sigma=10)
contour_data = contour_data/np.max(contour_data)*25
contour_plot = axes[2].contour(contour_data, levels=[5, 10, 15, 20], cmap=new_cmap, linewidths=1.2)
# axes[2].clabel(contour_plot, fmt='%i %%', fontsize=5)
new_cmap = truncate_colormap(cmap, 0, 1)
mappable = matplotlib.cm.ScalarMappable(norm=None, cmap=new_cmap)
ticks = [0.125, 0.375, 0.625, 0.875]
cax = axes[2].inset_axes([0.1, -0.1, 0.8, 0.05])
# cax = axes[2].inset_axes([1.05, 0, 0.05, 1])
orientation = "horizontal"
# orientation = "vertical"
cbar = plt.colorbar(mappable, orientation=orientation, ticks=ticks,
                    ax=axes[2], cax=cax,
                    # label="Mass concentration Ta (wt\\%)"
                    )
cbar.ax.set_xticklabels(["5\\%", "10\\%", "15\\%", "20\\%"])

plt.subplots_adjust(wspace=0.05, hspace=0.05)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# Create a Rectangle patch and add the patch to the ax
x0 = 1500
y0 = 900
h = 300
w = 500
xy = (int(x0 - w / 2), int(y0 - h / 2))
rect = matplotlib.patches.Rectangle(xy, w, h, linewidth=1.5, edgecolor=cmap(0.75), facecolor='none')
axes[0].add_patch(rect)

n, m = np.shape(entire_inclusion)
print(n, m)
xlen = m * 0.2385
x0 = m * 0.729
y0 = n * 13 / 15
yheight = n / 15
text = r"100 $\mu{}m$"
linewidth = 0.5
draw_scalebar(axes[0], x0, xlen, y0, yheight, text, linewidth)
xlen = m * 0.17
x0 = m * 0.77
text = r"10 $\mu{}m$"
draw_scalebar(axes[1], x0, xlen, y0, yheight, text, linewidth)
N = n
n, m = np.shape(density)
xlen = m * 0.2 * 0.9
x0 = m * 0.76
y0 = n * 13 / 15
yheight = n / 15
draw_scalebar(axes[2], x0, xlen, y0, yheight, text, linewidth, color="w")

for ax, letter, n in zip(axes, ["A", "B", "C"], (N, N, n)):
    pos = n/10
    fact = 1.5
    ax.annotate(letter, xy=(pos, pos*fact), xytext=(pos, pos*fact), horizontalalignment="center", verticalalignment="center",
                color="w", fontsize=16)

# name = "SEM_and_EDX.pdf"
name = "SEM_and_EDX_MDPI.pdf"

plot_path = os.path.join(cwd, "..", "figures", name)
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")

plot_path = os.path.join(cwd, "..", "..", "..", "Latex Documents", "ELO Image Analysis", "Images", name)
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")
