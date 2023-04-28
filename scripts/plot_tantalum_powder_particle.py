import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pickle
import os
import skimage
from PIL import Image


cwd = os.getcwd()

params = {'text.latex.preamble': r"\usepackage{lmodern}",
          'text.usetex': True,
          'font.size': 10,
          'font.family': 'lmodern'}
matplotlib.rcParams.update(params)


data_folder = os.path.join(cwd, "..", "data", "SEM TNM powder", "")
titanium = pd.read_csv(data_folder + "Ti.csv", header=None).dropna(axis=1)
aluminium = pd.read_csv(data_folder + "Al.csv", header=None).dropna(axis=1)
tantalum = pd.read_csv(data_folder + "Ta.csv", header=None).dropna(axis=1)
image_path = os.path.join(cwd, "..", "images", "SEM_image_of_TNM_powder_with_Tantalum.tif")
SEM = np.array(Image.open(image_path))
SEM = skimage.exposure.adjust_gamma(SEM, 2)

cutoff = 100
SEM = SEM[cutoff:-cutoff*3, :]

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(3.49, 3.49)

cmap = "gray_r"
ax.imshow(SEM, cmap=cmap)
ax.set_xticks([])
ax.set_yticks([])

n, m, _ = np.shape(SEM)
scalebar_length_200_micrometer = m * 0.2385
x0 = m * 0.7
x1 = x0 + scalebar_length_200_micrometer
y0 = n * 13 / 15
y1 = n * 14 / 15
x_scalebar = np.array([x0, x1])
y_scalebar = np.array([y0, y1])
unit_array = np.array([1, 1])

linewidth = 0.5
ax.plot(x_scalebar, unit_array*np.mean(y_scalebar), color="w", linewidth=linewidth)
ax.plot(unit_array*x_scalebar[0], y_scalebar, color="w", linewidth=linewidth)
ax.plot(unit_array*x_scalebar[1], y_scalebar, color="w", linewidth=linewidth)
xy = (np.mean(x_scalebar), 1.5*y_scalebar[0]-0.5*y_scalebar[1])
ax.annotate(r"200 $\mu{}m$",
            xy=xy,
            xytext=xy,
            horizontalalignment="center",
            verticalalignment="center",
            color="w")

plot_path = os.path.join(cwd, "..", "figures", "TNM_powder_with_Tantalum.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")

plot_path = os.path.join(cwd, "..", "..", "..", "Latex Documents", "ELO Image Analysis", "Images", "TNM_powder_with_Tantalum.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")