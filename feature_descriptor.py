import numpy as np
from skimage.feature import hog

# import time


def describe(
    feature,
    orientations_g=9,
    pixels_per_cell_g=(16, 16),
    cells_per_block_g=(4, 4),
    orientations_c=5,
    pixels_per_cell_c=(16, 16),
    cells_per_block_c=(4, 4),
):
    # start1 = time.perf_counter()
    grad = hog(
        feature,
        orientations=orientations_g,
        pixels_per_cell=pixels_per_cell_g,
        cells_per_block=cells_per_block_g,
        visualize=False,
        multichannel=False,
        transform_sqrt=True,
    )
    # end1 = time.perf_counter()
    # start2 = time.perf_counter()
    val = hov(
        feature,
        orientations=orientations_c,
        pixels_per_cell=pixels_per_cell_c,
        cells_per_block=cells_per_block_c,
        visualize=False,
        transform_sqrt=True,
    )
    # end2 = time.perf_counter()
    # print("time taken hog: {} s, hov: {} s".format(end1 - start1, end2 - start1))
    return np.concatenate((grad, val))


def _hog_normalize_block(block, method, eps=1e-5):
    if method == "L1":
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == "L1-sqrt":
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == "L2":
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif method == "L2-Hys":
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    else:
        raise ValueError("Selected block normalization method is invalid.")
    return out


# adapted from https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_hog.py
def hov(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    block_norm="L2-Hys",
    visualize=False,
    transform_sqrt=False,
    feature_vector=True,
):
    """Extract Histogram of Values (HOV) for a given image.
    Compute a Histogram of Oriented Gradients (HOG) by
        1. (optional) global image normalization
        2. computing the gradient image in `row` and `col`
        3. computing gradient histograms
        4. normalizing across blocks
        5. flattening into a feature vector
    Parameters
    ----------
    image : (M, N[, C]) ndarray
        Input image.
    orientations : int, optional
        Number of orientation bins.
    pixels_per_cell : 2-tuple (int, int), optional
        Size (in pixels) of a cell.
    cells_per_block : 2-tuple (int, int), optional
        Number of cells in each block.
    block_norm : str {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}, optional
        Block normalization method:
        ``L1``
           Normalization using L1-norm.
        ``L1-sqrt``
           Normalization using L1-norm, followed by square root.
        ``L2``
           Normalization using L2-norm.
        ``L2-Hys``
           Normalization using L2-norm, followed by limiting the
           maximum values to 0.2 (`Hys` stands for `hysteresis`) and
           renormalization using L2-norm. (default)
           For details, see [3]_, [4]_.
    visualize : bool, optional
        Also return an image of the HOG.  For each cell and orientation bin,
        the image contains a line segment that is centered at the cell center,
        is perpendicular to the midpoint of the range of angles spanned by the
        orientation bin, and has intensity proportional to the corresponding
        histogram value.
    transform_sqrt : bool, optional
        Apply power law compression to normalize the image before
        processing. DO NOT use this if the image contains negative
        values. Also see `notes` section below.
    feature_vector : bool, optional
        Return the data as a feature vector by calling .ravel() on the result
        just before returning.
    multichannel : boolean, optional
        If True, the last `image` dimension is considered as a color channel,
        otherwise as spatial.
    Returns
    -------
    out : (n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, n_orient) ndarray
        HOG descriptor for the image. If `feature_vector` is True, a 1D
        (flattened) array is returned.
    hog_image : (M, N) ndarray, optional
        A visualisation of the HOG image. Only provided if `visualize` is True.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    .. [2] Dalal, N and Triggs, B, Histograms of Oriented Gradients for
           Human Detection, IEEE Computer Society Conference on Computer
           Vision and Pattern Recognition 2005 San Diego, CA, USA,
           https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf,
           :DOI:`10.1109/CVPR.2005.177`
    .. [3] Lowe, D.G., Distinctive image features from scale-invatiant
           keypoints, International Journal of Computer Vision (2004) 60: 91,
           http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf,
           :DOI:`10.1023/B:VISI.0000029664.99615.94`
    .. [4] Dalal, N, Finding People in Images and Videos,
           Human-Computer Interaction [cs.HC], Institut National Polytechnique
           de Grenoble - INPG, 2006,
           https://tel.archives-ouvertes.fr/tel-00390303/file/NavneetDalalThesis.pdf
    Notes
    -----
    The presented code implements the HOG extraction method from [2]_ with
    the following changes: (I) blocks of (3, 3) cells are used ((2, 2) in the
    paper; (II) no smoothing within cells (Gaussian spatial window with sigma=8pix
    in the paper); (III) L1 block normalization is used (L2-Hys in the paper).
    Power law compression, also known as Gamma correction, is used to reduce
    the effects of shadowing and illumination variations. The compression makes
    the dark regions lighter. When the kwarg `transform_sqrt` is set to
    ``True``, the function computes the square root of each color channel
    and then applies the hog algorithm to the image.
    """
    image = np.atleast_2d(image)

    """
    The first stage applies an optional global image normalization
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each color channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    if transform_sqrt:
        image = np.sqrt(image)

    """
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    color channel is used, which provides color invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    if image.dtype.kind == "u":
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers
        image = image.astype("float")

    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    s_row, s_col = image.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations))

    for y in range(n_cells_col):
        for x in range(n_cells_row):
            sub = image[y * c_row : (y + 1) * c_row, x * c_col : (x + 1) * c_col]
            orientation_histogram[x, y, :] = np.histogram(sub, bins=orientations)[0] / (
                c_row * c_col
            )

    # now compute the histogram for each cell
    hog_image = None

    if visualize:
        from skimage.feature import draw

        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(orientations)
        # set dr_arr, dc_arr to correspond to midpoints of orientation bins
        orientation_bin_midpoints = np.pi * (orientations_arr + 0.5) / orientations
        dr_arr = radius * np.sin(orientation_bin_midpoints)
        dc_arr = radius * np.cos(orientation_bin_midpoints)
        hog_image = np.zeros((s_row, s_col), dtype=float)
        for r in range(n_cells_row):
            for c in range(n_cells_col):
                for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                    centre = tuple([r * c_row + c_row // 2, c * c_col + c_col // 2])
                    rr, cc = draw.line(
                        int(centre[0] - dc),
                        int(centre[1] + dr),
                        int(centre[0] + dc),
                        int(centre[1] - dr),
                    )
                    hog_image[rr, cc] += orientation_histogram[r, c, o]

    """
    The fourth stage computes normalization, which takes local groups of
    cells and contrast normalizes their overall responses before passing
    to next stage. Normalization introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalize each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalizations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalizations. This may seem redundant but it improves the performance.
    We refer to the normalized block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros(
        (n_blocks_row, n_blocks_col, b_row, b_col, orientations)
    )

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r : r + b_row, c : c + b_col, :]
            normalized_blocks[r, c, :] = _hog_normalize_block(block, method=block_norm)

    """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    if feature_vector:
        normalized_blocks = normalized_blocks.ravel()

    if visualize:
        return normalized_blocks, hog_image
    else:
        return normalized_blocks
