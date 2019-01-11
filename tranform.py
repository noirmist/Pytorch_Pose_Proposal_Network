# not using because of it's development for python3
#from __future__ import division

import math
import random
import PIL
import numpy as np
import warnings

import cv2

def _resize(img, size, interpolation):
    img = img.transpose((1, 2, 0))
    if interpolation == PIL.Image.NEAREST:
	cv_interpolation = cv2.INTER_NEAREST
    elif interpolation == PIL.Image.BILINEAR:
	cv_interpolation = cv2.INTER_LINEAR
    elif interpolation == PIL.Image.BICUBIC:
	cv_interpolation = cv2.INTER_CUBIC
    elif interpolation == PIL.Image.LANCZOS:
	cv_interpolation = cv2.INTER_LANCZOS4
    H, W = size
    img = cv2.resize(img, dsize=(W, H), interpolation=cv_interpolation)

    # If input is a grayscale image, cv2 returns a two-dimentional array.
    if len(img.shape) == 2:
	img = img[:, :, np.newaxis]
    return img.transpose((2, 0, 1))

def resize(img, size, interpolation=PIL.Image.BILINEAR):
    """Resize image to match the given shape.

    This method uses :mod:`cv2` or :mod:`PIL` for the backend.
    If :mod:`cv2` is installed, this function uses the implementation in
    :mod:`cv2`. This implementation is faster than the implementation in
    :mod:`PIL`. Under Anaconda environment,
    :mod:`cv2` can be installed by the following command.

    .. code::

        $ conda install -c menpo opencv3=3.2.0

    Args:
        img (~numpy.ndarray): An array to be transformed.
            This is in CHW format and the type should be :obj:`numpy.float32`.
        size (tuple): This is a tuple of length 2. Its elements are
            ordered as (height, width).
        interpolation (int): Determines sampling strategy. This is one of
            :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BILINEAR`,
            :obj:`PIL.Image.BICUBIC`, :obj:`PIL.Image.LANCZOS`.
            Bilinear interpolation is the default strategy.

    Returns:
        ~numpy.ndarray: A resize array in CHW format.

    """
    img = _resize(img, size, interpolation)
    return img

def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.
    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.
    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):
        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.
        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.
        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.
    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


def flip_point(point, size, y_flip=False, x_flip=False):
    """Modify points according to image flips.
    Args:
        point (~numpy.ndarray): Points in the image.
            The shape of this array is :math:`(P, 2)`. :math:`P` is the number
            of points in the image.
            The last dimension is composed of :math:`y` and :math:`x`
            coordinates of the points.
        size (tuple): A tuple of length 2. The height and the width
            of the image, which is associated with the points.
        y_flip (bool): Modify points according to a vertical flip of
            an image.
        x_flip (bool): Modify keypoipoints according to a horizontal flip of
            an image.
    Returns:
        ~numpy.ndarray:
        Points modified according to image flips.
    """
    H, W = size
    point = point.copy()
    if y_flip:
        point[:, 0] = H - point[:, 0]
    if x_flip:
        point[:, 1] = W - point[:, 1]
    return point



def scale(img, size, fit_short=True, interpolation=PIL.Image.BILINEAR):
    """Rescales the input image to the given "size".

    When :obj:`fit_short == True`, the input image will be resized so that
    the shorter edge will be scaled to length :obj:`size` after
    resizing. For example, if the height of the image is larger than
    its width, image will be resized to (size * height / width, size).

    Otherwise, the input image will be resized so that
    the longer edge will be scaled to length :obj:`size` after
    resizing.

    Args:
        img (~numpy.ndarray): An image array to be scaled. This is in
            CHW format.
        size (int): The length of the smaller edge.
        fit_short (bool): Determines whether to match the length
            of the shorter edge or the longer edge to :obj:`size`.
        interpolation (int): Determines sampling strategy. This is one of
            :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BILINEAR`,
            :obj:`PIL.Image.BICUBIC`, :obj:`PIL.Image.LANCZOS`.
            Bilinear interpolation is the default strategy.

    Returns:
        ~numpy.ndarray: A scaled image in CHW format.

    """
    _, H, W = img.shape

    # If resizing is not necessary, return the input as is.
    if fit_short and ((H <= W and H == size) or (W <= H and W == size)):
        return img
    if not fit_short and ((H >= W and H == size) or (W >= H and W == size)):
        return img

    if fit_short:
        if H < W:
            out_size = (size, int(size * W / H))
        else:
            out_size = (int(size * H / W), size)

    else:
        if H < W:
            out_size = (int(size * H / W), size)
        else:
            out_size = (size, int(size * W / H))

    return resize(img, out_size, interpolation)

def translate_point(point, y_offset=0, x_offset=0):
    """Translate points.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the top left point of the image
    to the coordinate :math:`(y, x) = (y_{offset}, x_{offset})`.

    Args:
        point (~numpy.ndarray): Points in the image.
            The shape of this array is :math:`(P, 2)`. :math:`P` is the number
            of points in the image.
            The last dimension is composed of :math:`y` and :math:`x`
            coordinates of the points.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    Returns:
        ~numpy.ndarray:
        Points modified translation of an image.

    """

    out_point = point.copy()

    out_point[:, 0] += y_offset
    out_point[:, 1] += x_offset

    return out_point


def random_sized_crop(img,
                      scale_ratio_range=(0.08, 1),
                      aspect_ratio_range=(3 / 4, 4 / 3),
                      return_param=False, copy=False):
    """Crop an image to random size and aspect ratio.

    The size :math:`(H_{crop}, W_{crop})` and the left top coordinate
    :math:`(y_{start}, x_{start})` of the crop are calculated as follows:

    + :math:`H_{crop} = \\lfloor{\\sqrt{s \\times H \\times W \
        \\times a}}\\rfloor`
    + :math:`W_{crop} = \\lfloor{\\sqrt{s \\times H \\times W \
        \\div a}}\\rfloor`
    + :math:`y_{start} \\sim Uniform\\{0, H - H_{crop}\\}`
    + :math:`x_{start} \\sim Uniform\\{0, W - W_{crop}\\}`
    + :math:`s \\sim Uniform(s_1, s_2)`
    + :math:`b \\sim Uniform(a_1, a_2)` and \
        :math:`a = b` or :math:`a = \\frac{1}{b}` in 50/50 probability.

    Here, :math:`s_1, s_2` are the two floats in
    :obj:`scale_ratio_range` and :math:`a_1, a_2` are the two floats
    in :obj:`aspect_ratio_range`.
    Also, :math:`H` and :math:`W` are the height and the width of the image.
    Note that :math:`s \\approx \\frac{H_{crop} \\times W_{crop}}{H \\times W}`
    and :math:`a \\approx \\frac{H_{crop}}{W_{crop}}`.
    The approximations come from flooring floats to integers.

    .. note::

        When it fails to sample a valid scale and aspect ratio for ten
        times, it picks values in a non-uniform way.
        If this happens, the selected scale ratio can be smaller
        than :obj:`scale_ratio_range[0]`.

    Args:
        img (~numpy.ndarray): An image array. This is in CHW format.
        scale_ratio_range (tuple of two floats): Determines
            the distribution from which a scale ratio is sampled.
            The default values are selected so that the area of the crop is
            8~100% of the original image. This is the default
            setting used to train ResNets in Torch style.
        aspect_ratio_range (tuple of two floats): Determines
            the distribution from which an aspect ratio is sampled.
            The default values are
            :math:`\\frac{3}{4}` and :math:`\\frac{4}{3}`, which
            are also the default setting to train ResNets in Torch style.
        return_param (bool): Returns parameters if :obj:`True`.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns only the cropped image.

        If :obj:`return_param = True`,
        returns a tuple of cropped image and :obj:`param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_slice** (*slice*): A slice used to crop the input image.\
            The relation below holds together with :obj:`x_slice`.
        * **x_slice** (*slice*): Similar to :obj:`y_slice`.

            .. code::

                out_img = img[:, y_slice, x_slice]

        * **scale_ratio** (float): :math:`s` in the description (see above).
        * **aspect_ratio** (float): :math:`a` in the description.

    """
    _, H, W = img.shape
    scale_ratio, aspect_ratio =\
        _sample_parameters(
            (H, W), scale_ratio_range, aspect_ratio_range)

    H_crop = int(math.floor(np.sqrt(scale_ratio * H * W * aspect_ratio)))
    W_crop = int(math.floor(np.sqrt(scale_ratio * H * W / aspect_ratio)))
    y_start = random.randint(0, H - H_crop)
    x_start = random.randint(0, W - W_crop)
    y_slice = slice(y_start, y_start + H_crop)
    x_slice = slice(x_start, x_start + W_crop)

    img = img[:, y_slice, x_slice]

    if copy:
        img = img.copy()
    if return_param:
        params = {'y_slice': y_slice, 'x_slice': x_slice,
                  'scale_ratio': scale_ratio, 'aspect_ratio': aspect_ratio}
        return img, params
    else:
        return img



def _sample_parameters(size, scale_ratio_range, aspect_ratio_range):
    H, W = size
    for _ in range(10):
        aspect_ratio = random.uniform(
            aspect_ratio_range[0], aspect_ratio_range[1])
        if random.uniform(0, 1) < 0.5:
            aspect_ratio = 1 / aspect_ratio
        # This is determined so that relationships "H - H_crop >= 0" and
        # "W - W_crop >= 0" are always satisfied.
        scale_ratio_max = min((scale_ratio_range[1],
                               H / (W * aspect_ratio),
                               (aspect_ratio * W) / H))

        scale_ratio = random.uniform(
            scale_ratio_range[0], scale_ratio_range[1])
        if scale_ratio_range[0] <= scale_ratio <= scale_ratio_max:
            return scale_ratio, aspect_ratio

    # This scale_ratio is outside the given range when
    # scale_ratio_max < scale_ratio_range[0].
    scale_ratio = random.uniform(
        min((scale_ratio_range[0], scale_ratio_max)), scale_ratio_max)

    return scale_ratio, aspect_ratio


def resize_point(point, in_size, out_size):
    """Adapt point coordinates to the rescaled image space.

    Args:
        point (~numpy.ndarray): Points in the image.
            The shape of this array is :math:`(P, 2)`. :math:`P` is the number
            of points in the image.
            The last dimension is composed of :math:`y` and :math:`x`
            coordinates of the points.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Points rescaled according to the given image shapes.

    """
    point = point.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    point[:, 0] = y_scale * point[:, 0]
    point[:, 1] = x_scale * point[:, 1]
    return point

def random_distort(
        img,
        brightness_delta=32,
        contrast_low=0.5, contrast_high=1.5,
        saturation_low=0.5, saturation_high=1.5,
        hue_delta=18):
    """A color related data augmentation used in SSD.

    This function is a combination of four augmentation methods:
    brightness, contrast, saturation and hue.

    * brightness: Adding a random offset to the intensity of the image.
    * contrast: Multiplying the intensity of the image by a random scale.
    * saturation: Multiplying the saturation of the image by a random scale.
    * hue: Adding a random offset to the hue of the image randomly.

    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_.

    Note that this function requires :mod:`cv2`.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        img (~numpy.ndarray): An image array to be augmented. This is in
            CHW and RGB format.
        brightness_delta (float): The offset for saturation will be
            drawn from :math:`[-brightness\_delta, brightness\_delta]`.
            The default value is :obj:`32`.
        contrast_low (float): The scale for contrast will be
            drawn from :math:`[contrast\_low, contrast\_high]`.
            The default value is :obj:`0.5`.
        contrast_high (float): See :obj:`contrast_low`.
            The default value is :obj:`1.5`.
        saturation_low (float): The scale for saturation will be
            drawn from :math:`[saturation\_low, saturation\_high]`.
            The default value is :obj:`0.5`.
        saturation_high (float): See :obj:`saturation_low`.
            The default value is :obj:`1.5`.
        hue_delta (float): The offset for hue will be
            drawn from :math:`[-hue\_delta, hue\_delta]`.
            The default value is :obj:`18`.

    Returns:
        An image in CHW and RGB format.

    """

    cv_img = img[::-1].transpose((1, 2, 0)).astype(np.uint8)

    def convert(img, alpha=1, beta=0):
        img = img.astype(float) * alpha + beta
        img[img < 0] = 0
        img[img > 255] = 255
        return img.astype(np.uint8)

    def brightness(cv_img, delta):
        if random.randrange(2):
            return convert(
                cv_img,
                beta=random.uniform(-delta, delta))
        else:
            return cv_img

    def contrast(cv_img, low, high):
        if random.randrange(2):
            return convert(
                cv_img,
                alpha=random.uniform(low, high))
        else:
            return cv_img

    def saturation(cv_img, low, high):
        if random.randrange(2):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            cv_img[:, :, 1] = convert(
                cv_img[:, :, 1],
                alpha=random.uniform(low, high))
            return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
        else:
            return cv_img

    def hue(cv_img, delta):
        if random.randrange(2):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            cv_img[:, :, 0] = (
                cv_img[:, :, 0].astype(int) +
                random.randint(-delta, delta)) % 180
            return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
        else:
            return cv_img

    cv_img = brightness(cv_img, brightness_delta)

    if random.randrange(2):
        cv_img = contrast(cv_img, contrast_low, contrast_high)
        cv_img = saturation(cv_img, saturation_low, saturation_high)
        cv_img = hue(cv_img, hue_delta)
    else:
        cv_img = saturation(cv_img, saturation_low, saturation_high)
        cv_img = hue(cv_img, hue_delta)
        cv_img = contrast(cv_img, contrast_low, contrast_high)

    return cv_img.astype(np.float32).transpose((2, 0, 1))[::-1]
