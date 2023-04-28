import numpy as np
import skimage
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
path = os.path.join(cwd, "..", "images", "ELO Material Contrast images", "Real Images", "K7_39494.JPG")
image = plt.imread(path)


path = os.path.join(cwd, "..", "images", "ELO Material Contrast images", "150kV_polished", "large_area", "11-53-28_2mA_large_area_edited.png")
ELO = plt.imread(path)

n, m, c = np.shape(image)
crop = int((m - n) / 2)
image = image[:, crop:-crop]





src = np.array([[0, 0], [0, n], [n, n], [n, 0]])
dst = np.array([[527, 680], [417, 2625], [2660, 2650], [2565, 680]])
dst = np.array([[400, 550], [300, 2800], [2800, 2800], [2700, 550]])

tform3 = skimage.transform.ProjectiveTransform()
tform3.estimate(src, dst)
warped = skimage.transform.warp(image, tform3, output_shape=(n, n))

zoom = int(n * 0.14)
zoomed_image = skimage.transform.resize(warped[zoom:-zoom, zoom:-zoom], (n, n), anti_aliasing=False)

zoom = 600
x = 25
y = 15
zoomed_image = skimage.transform.resize(zoomed_image[zoom-y:-zoom-y, zoom-x:-zoom-x], (n, n), anti_aliasing=False)

ELO = skimage.transform.resize(ELO, (n, n), anti_aliasing=False)
x = -15
y = 0
ELO = skimage.transform.resize(ELO[zoom-y:-y-zoom, zoom-x:-zoom-x], (n, n), anti_aliasing=False)


# fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
# axes = axes.flatten()

#axes[0].imshow(image)
# for a in axes:
#     a.axis('off')

# fig, ax = plt.subplots(1, 3, figsize=(18, 6))


# for a in ["A", "B", "C"]:
#     text.append(a)
#
# for i in range(12):
#     R = 840
#     phi = np.pi * (0.5 - i / 6)
#     coord = (n/2 + R * np.cos(phi), n/2 + R * np.sin(phi))
#     r = 135
#     circle = plt.Circle(coord, r, edgecolor="k", facecolor="w")
#     ax[0].add_patch(circle)
#     ax[0].text(coord[0], coord[1], text[i], horizontalalignment='center', verticalalignment='center', fontsize=16)
#
# # Middle circle
# circle = plt.Circle((n/2, n/2), 20, edgecolor="k", facecolor="w")
# ax[0].add_patch(circle)
#
# # Right-hand circle
# circle = plt.Circle((n/2 + 1400, n/2), 10, edgecolor="k", facecolor="w")
# ax[0].add_patch(circle)
#
# ax[0].plot([n/2 - R, n/2], [n/2 + R * 1.3, n/2 + R * 1.3], color="k")
# ax[0].plot([n/2 - R, n/2 - R], [n/2 + R * 1.25, n/2 + R * 1.35], color="k")
# ax[0].plot([n/2, n/2], [n/2 + R * 1.25, n/2 + R * 1.35], color="k")
#
# ax[0].text(n/2 - R/2, n/2 + R * 1.5, "25 mm", horizontalalignment='center', verticalalignment='center', fontsize=16)
#
# ax[0].plot([n/2 - r, n/2 + r], [n/2 - R * 1.3, n/2 - R * 1.3], color="k")
# ax[0].plot([n/2 - r, n/2 - r], [n/2 - R * 1.25, n/2 - R * 1.35], color="k")
# ax[0].plot([n/2 + r, n/2 + r], [n/2 - R * 1.25, n/2 - R * 1.35], color="k")
#
# ax[0].text(n/2, n/2 - R * 1.5, "8 mm", horizontalalignment='center', verticalalignment='center', fontsize=16)
#
# ax[0].axis([0, n, 0, n])
# ax[0].set_aspect("equal", "box")
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
#
#
# ax[1].imshow(zoomed_image)
# ax[2].imshow(ELO)
#
# ax[2].plot([n/2 - r, n/2 + r], [n/2 + R * 1.3, n/2 + R * 1.3], color="w")
# ax[2].plot([n/2 - r, n/2 - r], [n/2 + R * 1.25, n/2 + R * 1.35], color="w")
# ax[2].plot([n/2 + r, n/2 + r], [n/2 + R * 1.25, n/2 + R * 1.35], color="w")
#
# ax[2].text(n/2, n/2 + R * 1.5, "180 pxs", horizontalalignment='center', verticalalignment='center', fontsize=16, color="w")
#
# for a in ax[1:]:
#     a.axis('off')
#
# plt.tight_layout()
#
# plt.savefig("Materials_plate_3_figures.png")

fig, ax = plt.subplots(1, 2)
# fig, ax = plt.subplots(2, 1)

plt.rcParams['text.latex.preamble'] = r"\usepackage{lmodern}"
params = {'text.usetex': True,
          'font.size': 10,
          'font.family': 'lmodern'}
plt.rcParams.update(params)


fig.set_size_inches(3.49, 3.49)

ax[0].imshow(zoomed_image[::-1, :])
ax[1].imshow(ELO)

text = [str(i+1) for i in range(9)]
for a in ["A", "B", "C"]:
    text.append(a)

fontsize = 7
linewidth = 0.75
for i in range(12):
    R = 1020
    phi = np.pi * (0.5 - i / 6)
    coord = (n/2 + R * np.cos(phi), n/2 + R * np.sin(phi))
    r = 170
    circle = plt.Circle(coord, r, edgecolor="k", fill=False, linestyle="--", linewidth=linewidth)
    ax[0].add_artist(circle)
    ax[0].text(coord[0], coord[1], text[i], horizontalalignment='center', verticalalignment='center', fontsize=fontsize, fontweight="bold")

ax[0].plot([n/2 - R, n/2], [n/2 + R * 1.3, n/2 + R * 1.3], color="k", linewidth=linewidth)
ax[0].plot([n/2 - R, n/2 - R], [n/2 + R * 1.25, n/2 + R * 1.35], color="k", linewidth=linewidth)
ax[0].plot([n/2, n/2], [n/2 + R * 1.25, n/2 + R * 1.35], color="k", linewidth=linewidth)

ax[0].text(n/2 + r, n/2 + R * 1.3, "25 mm", horizontalalignment='left', verticalalignment='center', fontsize=fontsize)

ax[0].plot([n/2 - r, n/2 + r], [n/2 - R * 1.3, n/2 - R * 1.3], color="k", linewidth=linewidth)
ax[0].plot([n/2 - r, n/2 - r], [n/2 - R * 1.25, n/2 - R * 1.35], color="k", linewidth=linewidth)
ax[0].plot([n/2 + r, n/2 + r], [n/2 - R * 1.25, n/2 - R * 1.35], color="k", linewidth=linewidth)

ax[0].text(n/2 + 2*r, n/2 - R * 1.3, "8 mm", horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
ax[0].text(r/2, r, "CAM", horizontalalignment='left', verticalalignment='center')

#ax[0].text(n/2, n/2 - R * 1.6, "8 mm", horizontalalignment='center', verticalalignment='center')

ax[0].axis([0, n, 0, n])
ax[0].set_aspect("equal", "box")


ax[1].plot([n/2 - r, n/2 + r], [n/2 + R * 1.3, n/2 + R * 1.3], color="w", linewidth=linewidth)
ax[1].plot([n/2 - r, n/2 - r], [n/2 + R * 1.25, n/2 + R * 1.35], color="w", linewidth=linewidth)
ax[1].plot([n/2 + r, n/2 + r], [n/2 + R * 1.25, n/2 + R * 1.35], color="w", linewidth=linewidth)

ax[1].text(n/2 + 2*r, n/2 + R * 1.3, "180 pixels", horizontalalignment='left', verticalalignment='center', color="w", fontsize=fontsize)
ax[1].text(r/2, n-r, "ELO", horizontalalignment='left', verticalalignment='center', color="w")

ax[0].axis('off')
ax[1].axis('off')


plt.tight_layout()

plot_path = os.path.join(cwd, "..", "figures", "Material_plate_horizontal.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")

plot_path = os.path.join(cwd, "..", "..", "..", "Latex Documents", "ELO Image Analysis", "Images", "Material_plate_horizontal.pdf")
plt.savefig(plot_path, dpi=1000, bbox_inches="tight")