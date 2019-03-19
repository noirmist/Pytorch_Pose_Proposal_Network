import os
import glob
import itertools
import shutil
import numpy as np
from PIL import Image


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def parse_size(text):
    w, h = text.split('x')
    w = float(w)
    h = float(h)
    if w.is_integer():
        w = int(w)
    if h.is_integer():
        h = int(h)
    return w, h


def parse_kwargs(args):
    if args == '':
        return {}

    kwargs = {}
    for arg in args.split(','):
        key, value = arg.split('=')
        kwargs[key] = value

    return kwargs


def save_files(result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir, 'src'))
    result_src_dir = os.path.join(result_dir, 'src')
    file_list = glob.glob('*.py') + glob.glob('*.sh') + glob.glob('*.ini')
    file_list = file_list + glob.glob('*.tsv') + glob.glob('*.txt') + glob.glob("*.ipynb")
    for file in file_list:
        shutil.copy(file, os.path.join(result_src_dir, os.path.basename(file)))
    return result_src_dir


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (string): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

# Visutalization
def show_landmarks(image, keypoints, bbox, fname):
    """Show image with keypoints"""
    #print("show_landmarks:", type(image), image.dtype)

    image = image.numpy().astype(np.uint8)
    image = np.array(image).transpose((1, 2, 0)) 
    bboxes = bbox.numpy()
    keypoints = keypoints.reshape(-1,2).numpy()
    
    fig = plt.figure()
    plt.imshow(image)

    # change 0 to nan
    x = keypoints[:,0]
    x[x==0] = np.nan

    y = keypoints[:,1]
    y[y==0] = np.nan

    for (cx1,cy1,w,h) in bboxes:
        rect = patches.Rectangle((cx1-w//2,cy1-h//2),w,h,linewidth=2,edgecolor='b',facecolor='none')
        plt.gca().add_patch(rect)

    plt.scatter(keypoints[:, 0], keypoints[:, 1], marker='.', c='r')
    plt.pause(0.0001)  # pause a bit so that plots are updated
    plt.savefig("aug_img/"+fname) 
    plt.close()

