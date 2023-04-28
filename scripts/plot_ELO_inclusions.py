import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pickle
import os
import skimage
from PIL import Image


def create_annulus_mask(h, w, center=None, inner_radius=None, outer_radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if inner_radius is None:
        inner_radius = 0
    if outer_radius is None: # use the smallest distance between the center and image walls
        inner_radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask_outer = dist_from_center <= outer_radius
    mask_inner = dist_from_center >= inner_radius
    mask = mask_inner * mask_outer
    return mask


cwd = os.getcwd()
preamble = '\n'.join([r"\usepackage[T1]{fontenc}",
                      r"\usepackage{lmodern}",
                      r"\usepackage{tikz}",
                      r"\usepackage[detect-all]{siunitx}"])

params = {'text.latex.preamble': preamble,
          'text.usetex': True,
          'font.size': 8,
          'font.family': 'serif',
          'font.serif': 'Times New Roman'}
#mpl.use('pgf')

params = {'text.latex.preamble': preamble,
          'text.usetex': True,
          'font.size': 10,
          'font.family': 'lmodern'}
mpl.rcParams.update(params)

#fig, axes = plt.subplots(2, 2)
fig, axes = plt.subplots(1, 4)
axes = axes.flatten()
#fig.set_size_inches(3.49, 3.49)
fig.set_size_inches(6, 1.3)


twinax = axes[3].twinx()
axes[3].set_yticks([])
axes[3] = twinax

path = os.path.join(cwd, "..", "images", "Real Time Inclusion Detection Experiment", "MetaData")
layers = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(".tif")]

inclusion_indices = [33, 51, 52, 88, 102, 105]
cube_coordinates = [[750, 1065], [1382, 750], [1075, 1065], [1075, 117], [1382, 430], [1075, 430]]
inclusion_coordinates = [[100, 41], [47, 103], [162, 54], [88, 107], [58, 73], [112, 125]]

for i in range(6):
    index = inclusion_indices[i]
    coord = cube_coordinates[i]
    raw_image = np.array(Image.open(layers[i]))
    image = (raw_image - 2 ** 15) / 2 ** 15 * 10
    m, n = np.shape(image)
    step = int(m/15)

    xmin = max(coord[0] - step, 0)
    xmax = min(coord[0] + step, m)
    ymin = max(coord[1] - step, 0)
    ymax = min(coord[1] + step, m)

    square_x = [xmin, xmin, xmax, xmax, xmin]
    square_y = [ymin, ymax, ymax, ymin, ymin]
    small_image = image[ymin:ymax, xmin:xmax]
    style = "|-|"
    widthA = 0
    widthB = 0.5
    bracketstyle = "|-|, widthB=0.5, widthA=0"
    arrowprops = dict(arrowstyle="->", color='k')
    #arrowprops = bracketstyle

    if i == 4:
        ax = axes[0]
        ax.imshow(image, cmap="gray")
        ax.plot(square_x, square_y, "C1")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(r"120 mm")
        ax.annotate('', xy=(1, 1.125), xycoords='axes fraction', xytext=(0.8, 1.125), arrowprops=arrowprops)
        ax.annotate('', xy=(0, 1.125), xycoords='axes fraction', xytext=(0.2, 1.125), arrowprops=arrowprops)
        #bracket = matplotlib.patches.FancyArrowPatch((10, 20), (30, 40), arrowstyle=bracketstyle)
        #ax.add_patch(bracket)
        ax.annotate("A", xy=(0.05, 0.05), xytext=(0.05, 0.05),
                    xycoords='axes fraction',
                    horizontalalignment="left", verticalalignment="bottom",
                    color="w", fontsize=16)

        ax = axes[1]
        ax.imshow(small_image, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(r"%i mm" % (120*(xmax-xmin)/n))
        ax.annotate('', xy=(1, 1.125), xycoords='axes fraction', xytext=(0.8, 1.125), arrowprops=arrowprops)
        ax.annotate('', xy=(0, 1.125), xycoords='axes fraction', xytext=(0.2, 1.125), arrowprops=arrowprops)
        ax.annotate("B", xy=(0.05, 0.05), xytext=(0.05, 0.05),
                    xycoords='axes fraction',
                    horizontalalignment="left", verticalalignment="bottom",
                    color="w", fontsize=16)

    gridsize = 15
    x = inclusion_coordinates[i][0]
    y = inclusion_coordinates[i][1]
    inclusion_image = small_image[y - gridsize:y + gridsize, x - gridsize:x + gridsize]
    inclusion_value = np.max(inclusion_image[10:-10, 10:-10])
    #surrounding_value = np.mean([np.median(inclusion_image[a:a+10, b:b+10]) for a, b in zip([0, 0, 0, 10, 10, 20, 20, 20], [0, 10, 20, 0, 20, 0, 10, 20])])
    cutoff = 40
    inclusion_cube = small_image[cutoff:-cutoff, cutoff:-cutoff]
    #nn, mm = np.shape(small_image)
    #ax.plot([cutoff, mm - cutoff, mm - cutoff, cutoff, cutoff], [cutoff, cutoff, mm - cutoff, mm - cutoff, cutoff])

    surrounding_value = np.median(inclusion_cube)

    middle_cube_value = np.median(image[680:820, 680:820])
    print(middle_cube_value)
    factor = middle_cube_value/surrounding_value

    # I = 2
    # a, b, c = (4.412, 9.636, -0.064)
    # estimated_Z_surrounding = (-b * np.log(2) / np.log((surrounding_value*factor/(I*a) - c)))**2
    # estimated_Z_inclusion = (-b * np.log(2) / np.log((inclusion_value*factor/(I*a) - c)))**2

    xmin = x - gridsize
    xmax = x + gridsize
    ymin = y - gridsize
    ymax = y + gridsize
    square_x = [xmin, xmin, xmax, xmax, xmin]
    square_y = [ymin, ymax, ymax, ymin, ymin]
    if i == 4:
        ax.plot(square_x, square_y, "C1")
        ax = axes[2]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(r"2.4 mm")
        # ax.set_title(r"\SI{800}{\micro\meter}")
        ax.annotate('', xy=(1, 1.125), xycoords='axes fraction', xytext=(0.8, 1.125), arrowprops=arrowprops)
        ax.annotate('', xy=(0, 1.125), xycoords='axes fraction', xytext=(0.2, 1.125), arrowprops=arrowprops)
        ax.annotate("C", xy=(0.05, 0.05), xytext=(0.05, 0.05),
                    xycoords='axes fraction',
                    horizontalalignment="left", verticalalignment="bottom",
                    color="w", fontsize=16)

        #ax.imshow(skimage.transform.rescale(skimage.filters.gaussian(inclusion_image, 0.65), (10, 10), order=0), cmap="gray")
        ax.imshow(inclusion_image, cmap="gray")
        max_row = np.max(inclusion_image[5:-5], axis=0)
        min_row = np.min(inclusion_image[5:-5], axis=0)

        ax = axes[3]
        ax.plot(max_row*factor, color="C1", alpha=1, zorder=2)
        ax.plot(min_row*factor, color="C1", alpha=1, zorder=2)
        ax.fill_between(np.arange(len(min_row)), min_row*factor, max_row*factor, color="C1", alpha=0.5, zorder=2)
        # for row in inclusion_image[10:-10]:
        #     ax.plot(row*factor, color="C1", alpha=0.25)
        ax.set_xticks([])
        ax.axis([0, gridsize*2, 1.65, 1.95])
        #ax.set_ylabel("Detector voltage [V]")
        ax.set_title("ELO voltage [V]")
        ax.annotate("D", xy=(0.05, 0.05), xytext=(0.05, 0.05),
                    xycoords='axes fraction',
                    horizontalalignment="left", verticalalignment="bottom",
                    color="k", fontsize=16)
    else:
        joke = "fun"
        # ax = axes[3]
        # max_row = np.max(inclusion_image[5:-5], axis=0)
        # min_row = np.min(inclusion_image[5:-5], axis=0)
        # ax.plot(max_row*factor, color="C0", linestyle="-", alpha=0.75, zorder=1)
        # ax.plot(min_row*factor, color="C0", linestyle="-", alpha=0.75, zorder=1)
        # ax.fill_between(np.arange(len(min_row)), min_row*factor, max_row*factor, color="C0", alpha=0.1, zorder=0)



axes[3].set_xlim(right=29)

plot_path = os.path.join(cwd, "..", "figures", "ELO_inclusions_MDPI.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")

plot_path = os.path.join(cwd, "..", "..", "..", "Latex Documents", "ELO Image Analysis", "Images", "ELO_inclusions_MDPI.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")
