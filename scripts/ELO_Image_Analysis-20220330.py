# Standard package for scientific computing. Multidimensional array object with several fast operations
import numpy as np

# Operating system dependent functionality.
import os

# Collection of algorithms for image processing. Sci kit-image
from skimage.io import imread
from skimage.filters import gaussian, threshold_otsu, sobel, threshold_local
from skimage.segmentation import watershed, flood_fill
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects, erosion, dilation, disk, closing
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.exposure import adjust_gamma


# Comprehensive library for creating visualizations in Python.
import matplotlib
import matplotlib.pyplot as plt


# Specify where to place the pop-up figure during plt.show()
# Function taken from:
# https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


cwd = os.getcwd()
layer_time = 60
first_loop = True
path_to_folder = os.path.join(cwd, "images", "Real Time Inclusion Detection Experiment", "Metadata")
paths_to_images = [os.path.join(path_to_folder, path) for path in os.listdir(path_to_folder) if path.endswith(".tif")]

for image_path in paths_to_images[1:]:
    print("\n" + 4*"========================" + "\n")
    print("Running ELO Image Analysis on %s" % image_path)

    n_rows = 2
    n_cols = 2
    fig_size = (9, 9)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size, sharex=True, sharey=True)
    axes = axes.ravel()
    fig.suptitle(image_path)

    images = []
    titles = []
    x_labels = []
    y_labels = []

    # Load image and add to list of images to plot
    image = plt.imread(image_path)
    image = image - np.min(image)
    N, M = np.shape(image)

    gamma = 2
    images.append(adjust_gamma(image, gamma=gamma))
    titles.append("Original ELO image (contrast enhanced)")
    x_labels.append("")
    y_labels.append("pixels")

    images.append(image)
    titles.append("ELO image with contours and defects")
    x_labels.append("")
    y_labels.append("")

    images.append(image)
    titles.append("ELO image with detected pores")
    x_labels.append("pixels")
    y_labels.append("pixels")

    # Use gaussian blurring to reduce influence of noise
    sigma = 1
    blurred_image = gaussian(image, sigma=sigma, preserve_range=True)

    # Find threshold dividing dark and light areas in image using Otsu's thresholding algorithm
    threshold = threshold_otsu(image)

    # Find edges in image using Sobel's edge detection algorithm
    edges = sobel(image)

    # Set markers used in the segmentation.
    # The darkest pixels get a value of 1 and the lightest get a value of 2. The rest 0.
    markers = np.zeros_like(image)
    image_range = np.max(image) - np.min(image)
    markers[image < np.min(image) + image_range*0.1] = 1
    # markers[10:-10, 10:-10] = 0
    markers[image > np.min(image) + image_range*0.8] = 2

    # Segment the dark area from the light area using watersheding.
    # The Sobel edges define the boundaries and the markers label the regions.
    segmentation = watershed(edges, markers)

    # Clean the segmentation from any small object (smaller than 400 pixels)s, e.g. support structures
    smallest_size = 400
    cleaned = remove_small_objects(segmentation > 1, smallest_size)

    # Find contours from segmentation
    contours = find_contours(cleaned, level=0.5)

    # Plot the contours on the ELO image.
    # Calculate length of the contour using Euclidean distance
    # Calculate area and centroid of the contour using Shoelace algorithm
    # Plot centroids as + markers
    contour_lengths = []
    contour_areas = []
    contour_centroids = []
    print("\n")
    for i, c in enumerate(contours):
        x = c[:, 1]
        y = c[:, 0]
        length = np.sum([np.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2) for i in range(len(x) - 1)])
        axes[1].plot(x, y, color="C%i" % i)

        # Shoelace algorithm
        determinant = [x[i] * y[i + 1] - y[i] * x[i + 1] for i in range(len(x) - 1)]
        x_pair = [x[i] + x[i + 1] for i in range(len(x) - 1)]
        y_pair = [y[i] + y[i + 1] for i in range(len(x) - 1)]
        Area = np.sum(determinant) / 2
        centroid_x = np.sum(np.array(determinant) * np.array(x_pair)) / (6 * Area)
        centroid_y = np.sum(np.array(determinant) * np.array(y_pair)) / (6 * Area)
        centroid = (int(centroid_x), int(centroid_y))
        axes[1].plot(centroid_x, centroid_y, marker="+", color="C%i" % i)

        contour_lengths.append(length)
        contour_areas.append(abs(Area))
        contour_centroids.append(centroid)

        print("Melt Contour [%i/%i]:     length = %i pixels    area = %i pixels    centroid = %s"
              % (i+1, len(contours), int(length), Area, str(centroid)))

    # Erode segmentation to avoid false detections in the boundary area
    eroded = erosion(cleaned, footprint=disk(10))

    # Detect the pixels that have a lower value than the Otsu threshold within the eroded segmentation
    defects = (blurred_image < threshold + image_range*0.1) * eroded

    # Dilate the defects and make them slightly bigger
    dilated_defects = dilation(defects, footprint=disk(5)) < 1

    # Find contours of defects
    defect_contours = find_contours(dilated_defects, level=0.5)

    # Plot the defect contours on the ELO image.
    # Calculate length of the defects using Euclidean distance
    # Calculate area and centroid of the defects using the Shoelace algorithm
    # Plot centroids as x markers
    print("\n")
    defect_area = 0
    for i, c in enumerate(defect_contours):
        x = c[:, 1]
        y = c[:, 0]
        length = np.sum([np.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2) for i in range(len(x) - 1)])
        axes[1].plot(x, y, color="r")

        # Shoelace algorithm
        determinant = [x[i] * y[i + 1] - y[i] * x[i + 1] for i in range(len(x) - 1)]
        x_pair = [x[i] + x[i + 1] for i in range(len(x) - 1)]
        y_pair = [y[i] + y[i + 1] for i in range(len(x) - 1)]
        Area = np.sum(determinant) / 2
        centroid_x = np.sum(np.array(determinant) * np.array(x_pair)) / (6 * Area)
        centroid_y = np.sum(np.array(determinant) * np.array(y_pair)) / (6 * Area)
        centroid = (int(centroid_x), int(centroid_y))
        axes[1].plot(centroid_x, centroid_y, marker="x", color="r")

        defect_area += abs(Area)

    print("%i defects found covering %.4f percent of the melt area" %
          (len(defect_contours), 100*defect_area/(np.sum(contour_areas)+1)))

    # Isolate features by inverting the image to make dark areas bright
    # and removeing the areas outside the contour and inside the defects
    gamma = 2
    features = image * eroded * dilated_defects

    # Detect the edges from the features
    sigma = 0.5
    canny_edges = canny(features/np.max(features), sigma=sigma)
    # img = canny_edges
    # axes[0].imshow(features, cmap="gray")
    # axes[2].imshow(img, cmap="gray")
    # axes[3].imshow(sobel(features), cmap="gray")
    # plt.show()
    # quit()

    # Plot the edges
    images.append(canny_edges)
    titles.append("Canny edges used to detect pores")
    x_labels.append("pixels")
    y_labels.append("")

    square = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    # Sometimes the pores are undetected because they don't have closed edges
    # closing the borders can help find them, but it also creates more false positives
    closed_edges = closing(canny_edges, footprint=square)

    cross = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])

    # Flood the figure. Once in the corner and once on each centroid.
    # All that should be left after the flood are the edge features that had closed borders
    flooded = flood_fill(canny_edges, footprint=cross, seed_point=(5,5), new_value=1)
    closed_flooded = flood_fill(closed_edges, footprint=cross, seed_point=(5,5), new_value=1)
    for centroid in contour_centroids:
        flooded = flood_fill(canny_edges, footprint=cross, seed_point=centroid, new_value=1)
        closed_flooded = flood_fill(closed_edges, footprint=cross, seed_point=centroid, new_value=1)

    # Find the remaining featrures that are circular, with small radii
    hough_radii = np.arange(2, 6, 1)
    min_x = 15
    min_y = 15
    hough_res = hough_circle(flooded, hough_radii)
    hough_res_closed = hough_circle(closed_flooded, hough_radii)
    hough_peaks = hough_circle_peaks(hough_res, hough_radii, min_xdistance=min_x, min_ydistance=min_y)
    hough_peaks_closed = hough_circle_peaks(hough_res_closed, hough_radii, min_xdistance=min_x, min_ydistance=min_y)

    # Calculate local thresholds to determine if the pores are gas pores of inclusions
    local_thresholds_small = threshold_local(image, 3)
    local_thresholds_big = threshold_local(image, 25)

    theta = np.linspace(0, 2 * np.pi, 24)
    gas_pores = 0
    inclusions = 0
    for peaks, linestyle in zip([hough_peaks, hough_peaks_closed], ["-", "--"]):
        _, center_x, center_y, radii = peaks
        for cx, cy, r in zip(center_x, center_y, radii):
            if local_thresholds_small[int(cy), int(cx)] * 0.95 > local_thresholds_big[int(cy), int(cx)]:
                color = "C3"
                inclusions += 1
            else:
                color = "C0"
                gas_pores += 1

            axes[2].plot(cx + np.cos(theta) * r, cy + np.sin(theta) * r, linestyle=linestyle, color=color)

    print("\n%i pores detected: %i gas pores and %i inclusions" % (gas_pores + inclusions, gas_pores, inclusions))

    for i in range(min(len(images), n_rows * n_cols)):
        image = images[i]
        title = titles[i]
        xlabel = x_labels[i]
        ylabel = y_labels[i]

        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(title)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)



    if not first_loop:
        plt.close(fig=old_fig)
    else:
        first_loop = False

    old_fig = fig
    move_figure(fig, 20, 20)
    plt.show(block=False)
    plt.pause(layer_time)