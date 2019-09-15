import tensorflow as tf
import tensorflow.contrib as tfcontrib


def shift_img(base_img, label_img, shift_range_pxls=(0, 0)):
    """This fn will perform the horizontal or vertical shift.

    Parameters
    ----------
    base_img : tensorflow.python.framework.ops.Tensor
        Tensor representation of input image
    label_img : tensorflow.python.framework.ops.Tensor
        Tensor representation of input image mask
    shift_range_pxls : tuple of 2 int, optional (default=(0, 0))
        Shift range in pixels for uniform distribution (width, height)

    Returns
    -------
    tuple of 2 tensorflow.python.framework.ops.Tensor
        Shifted images as tuple of Tensors (base_img, label_img)

    """
    if shift_range_pxls:
        width_shift_range = tf.random.uniform(
            [],
            -shift_range_pxls[1],
            shift_range_pxls[1],
        )
        height_shift_range = tf.random.uniform(
            [],
            -shift_range_pxls[0],
            shift_range_pxls[0],
        )
        # Translate both
        base_img = tfcontrib.image.translate(
            base_img,
            [width_shift_range, height_shift_range],
        )
        label_img = tfcontrib.image.translate(
            label_img,
            [width_shift_range, height_shift_range],
        )
    return base_img, label_img


def flip_img(base_img, label_img, horizontal_flip):
    """Flip the image horizontally along the central axis with a 0.5 probability

    Parameters
    ----------
    base_img : tensorflow.python.framework.ops.Tensor
        Tensor representation of input image
    label_img : tensorflow.python.framework.ops.Tensor
        Tensor representation of input image mask
    horizontal_flip : boolean
        Use True for flipping images

    Returns
    -------
    tuple of 2 tensorflow.python.framework.ops.Tensor
        Flipped images as tuple of Tensors (base_img, label_img)

    """
    if horizontal_flip:
        flip_prob = tf.random.uniform([], 0.0, 1.0)
        base_img, label_img = tf.cond(
            tf.less(flip_prob, 0.5),
            lambda: (
                tf.image.flip_left_right(base_img),
                tf.image.flip_left_right(label_img)
            ),
            lambda: (base_img, label_img),
        )
    return base_img, label_img


def rotate_img(base_img, label_img, angle_rad_range):
    """Rotate the image on angle in uniform(0, -angle_rad_range, angle_rad_range)

    Parameters
    ----------
    base_img : tensorflow.python.framework.ops.Tensor
        Tensor representation of input image
    label_img : tensorflow.python.framework.ops.Tensor
        Tensor representation of input image mask
    angle_rad_range : float
        Angle (radian) rotate range for uniform distribution

    Returns
    -------
    tuple of 2 tensorflow.python.framework.ops.Tensor
        Rotated images as tuple of Tensors (base_img, label_img)

    """
    if angle_rad_range:
        angle_rad = tf.random.uniform([], -angle_rad_range, angle_rad_range)
        base_img = tf.contrib.image.rotate(base_img, angle_rad)
        label_img = tf.contrib.image.rotate(label_img, angle_rad)
    return base_img, label_img


def augment(base_img,
            label_img,
            resize=None,  # Resize the image to some size e.g. [256, 256]
            scale=1,  # Scale image e.g. 1 / 255.
            hue_delta=0,  # Adjust the hue of an RGB image by random factor
            horizontal_flip=False,  # Random left right flip,
            shift_range_pxls=(0, 0),
            angle_rad_range=0):
    """Applies different augument methods for images

    Parameters
    ----------
    base_img : tensorflow.python.framework.ops.Tensor
        Tensor representation of input image
    label_img : tensorflow.python.framework.ops.Tensor
        Tensor representation of input image mask
    angle_rad_range : float
        Angle (radian) rotate range for uniform distribution

    Returns
    -------
    tuple of 2 tensorflow.python.framework.ops.Tensor
        Rotated images as tuple of Tensors (base_img, label_img)

    """
    if resize is not None:
        # Resize both images
        label_img = tf.image.resize(label_img, resize)
        base_img = tf.image.resize(base_img, resize)

    if hue_delta:
        base_img = tf.image.random_hue(base_img, hue_delta)

    base_img, label_img = flip_img(base_img, label_img, horizontal_flip)
    base_img, label_img = rotate_img(base_img, label_img, angle_rad_range)
    base_img, label_img = shift_img(base_img, label_img, shift_range_pxls)
    label_img = tf.cast(label_img, dtype=tf.float32) * scale
    base_img = tf.cast(base_img, dtype=tf.float32) * scale

    return base_img, label_img
