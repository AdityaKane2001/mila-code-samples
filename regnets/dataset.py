import math
import os
from datetime import datetime
from typing import Callable, List, Tuple, Type, Union

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

tfd = tfp.distributions


AUTO = tf.data.AUTOTUNE

_TFRECS_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "height": tf.io.FixedLenFeature([], tf.int64),
    "width": tf.io.FixedLenFeature([], tf.int64),
    "filename": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "synset": tf.io.FixedLenFeature([], tf.string),
}


class ImageNet:
    """Class for all ImageNet data-related functions, including TFRecord
    parsing along with augmentation transforms. TFRecords must follow the format
    given in _TFRECS_FORMAT. If not specified otherwise in `augment_fn` argument, following
    augmentations are applied to the dataset:
    - Color Jitter (random brightness, hue, saturation, contrast, flip)
        This augmentation is inspired by SimCLR (https://arxiv.org/abs/2002.05709).
        The strength parameter is set to 5, which controlsthe effect of augmentations.
    - Random rotate
    - Random crop and resize

    If `augment_fn` argument is not set to the string "default", it should be set to
    a callable object. That callable must take exactly two arguments: `image` and `target`
    and must return two values corresponding to the same.

    If `augment_fn` argument is 'val', then the images will be center cropped to 224x224.

    Args:
       cfg: regnety.config.config.PreprocessingConfig instance.
       no_aug: If True, overrides cfg and returns images as they are. Requires cfg object
            to determine batch_size, image_size, etc.
    """

    def __init__(self, cfg, no_aug=False):

        self.tfrecs_filepath = cfg.tfrecs_filepath
        self.batch_size = cfg.batch_size
        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size
        self.resize_pre_crop = cfg.resize_pre_crop
        self.augment_fn = cfg.augment_fn
        self.num_classes = cfg.num_classes
        self.color_jitter = cfg.color_jitter
        self.mixup = cfg.mixup
        self.mixup_alpha = cfg.mixup_alpha
        self.area_factor = cfg.area_factor
        self.no_aug = no_aug
        eigen_vals = tf.constant(
            [
                [0.2175, 0.0188, 0.0045],
                [0.2175, 0.0188, 0.0045],
                [0.2175, 0.0188, 0.0045],
            ]
        )
        self.eigen_vals = tf.stack([eigen_vals] * self.batch_size, axis=0)
        eigen_vecs = tf.constant(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        )
        self.eigen_vecs = tf.stack([eigen_vecs] * self.batch_size, axis=0)

        if (self.tfrecs_filepath is None) or (self.tfrecs_filepath == []):
            raise ValueError("List of TFrecords paths cannot be None or empty")

        if self.augment_fn == "default":
            self.default_augment = True
            self.val_augment = False
            self.strength = 5
            self.brightness_delta = self.strength * 0.1
            self.contrast_lower = 1 - 0.5 * (self.strength / 10.0)
            self.contrast_upper = 1 + 0.5 * (self.strength / 10.0)
            self.hue_delta = self.strength * 0.05
            self.saturation_lower = 1 - 0.5 * (self.strength / 10.0)
            self.saturation_upper = (1 - 0.5 * (self.strength / 10.0)) * 5
        elif self.augment_fn == "val":
            self.default_augment = False
            self.val_augment = True
            self.strength = -1
        else:
            self.default_augment = False
            self.val_augment = False
            self.strength = -1

    def decode_example(self, example_: tf.Tensor) -> dict:
        """Decodes an example to its individual attributes.

        Args:
            example: A TFRecord dataset example.

        Returns:
            Dict containing attributes from a single example. Follows
            the same names as _TFRECS_FORMAT.
        """

        example = tf.io.parse_example(example_, _TFRECS_FORMAT)
        image = tf.reshape(
            tf.io.decode_jpeg(example["image"]),
            (example["height"], example["width"], 3),
        )
        height = example["height"]
        width = example["width"]
        filename = example["filename"]
        label = example["label"]
        synset = example["synset"]
        return {
            "image": image,
            "height": height,
            "width": width,
            "filename": filename,
            "label": label,
            "synset": synset,
        }

    def _read_tfrecs(self) -> Type[tf.data.Dataset]:
        """Function for reading and loading TFRecords into a tf.data.Dataset.

        Args: None.

        Returns:
            A tf.data.Dataset instance.
        """

        files = tf.data.Dataset.list_files(self.tfrecs_filepath)
        ds = files.interleave(
            tf.data.TFRecordDataset, num_parallel_calls=AUTO, deterministic=False
        )

        ds = ds.map(self.decode_example, num_parallel_calls=AUTO)

        # ds = ds.map(self._one_hot_encode_example, num_parallel_calls=AUTO)
        # ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTO)
        return ds

    def _color_jitter(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Performs color jitter on the batch. It performs random brightness, hue, saturation,
        contrast and random left-right flip.

        Args:
            image: Batch of images to perform color jitter on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        aug_images = tf.image.random_brightness(image, self.brightness_delta)
        aug_images = tf.image.random_contrast(
            aug_images, self.contrast_lower, self.contrast_upper
        )
        aug_images = tf.image.random_hue(aug_images, self.hue_delta)
        aug_images = tf.image.random_saturation(
            aug_images, self.saturation_lower, self.saturation_upper
        )

        return aug_images, target

    def _inception_style_crop_batched(self, images, labels):
        """
        Applies inception style cropping

        Args:
            image: Batch of images to perform random rotation on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """
        # # Get target metrics
        area_ratio = tf.random.uniform((), minval=self.area_factor, maxval=1.0)

        aspect_ratio = tf.random.uniform((), minval=3.0 / 4.0, maxval=4.0 / 3.0)

        target_area = self.image_size**2 * area_ratio

        w = tf.cast(
            tf.clip_by_value(tf.round(tf.sqrt(target_area * aspect_ratio)), 0, 511),
            tf.int32,
        )

        h = tf.cast(
            tf.clip_by_value(tf.round(tf.sqrt(target_area / aspect_ratio)), 0, 511),
            tf.int32,
        )

        y0s = tf.random.uniform(
            (), minval=0, maxval=self.image_size - h + 1, dtype=tf.int32
        )
        x0s = tf.random.uniform(
            (), minval=0, maxval=self.image_size - w + 1, dtype=tf.int32
        )

        begins = [0, y0s, x0s, 0]
        sizes = [self.batch_size, h, w, 3]

        aug_images = tf.slice(images, begins, sizes)
        aug_images = tf.image.resize(aug_images, (224, 224))

        return aug_images, labels

    def _pca_jitter(self, image, target):
        """
        Applies PCA jitter to images.

        Args:
            image: Batch of images to perform random rotation on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        aug_images = tf.cast(image, tf.float32) / 255.0
        alpha = tf.random.normal((self.batch_size, 3), stddev=0.1)
        alpha = tf.stack([alpha, alpha, alpha], axis=1)
        rgb = tf.math.reduce_sum(alpha * self.eigen_vals * self.eigen_vecs, axis=2)
        rgb = tf.expand_dims(rgb, axis=1)
        rgb = tf.expand_dims(rgb, axis=1)

        aug_images = aug_images + rgb
        aug_images = aug_images * 255.0

        aug_images = tf.cast(tf.clip_by_value(aug_images, 0, 255), tf.uint8)

        return aug_images, target

    def random_flip(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Returns randomly flipped batch of images. Only horizontal flip
        is available

        Args:
            image: Batch of images to perform random rotation on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        aug_images = tf.image.random_flip_left_right(image)
        return aug_images, target

    def random_rotate(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Returns randomly rotated batch of images.

        Args:
            image: Batch of images to perform random rotation on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        angles = tf.random.uniform((self.batch_size,)) * (math.pi / 2.0)
        rotated = tfa.image.rotate(image, angles, fill_value=128.0)
        return rotated, target

    def random_crop(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """ "
        Returns random crops of images.

        Args:
            image: Batch of images to perform random crop on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        cropped = tf.image.random_crop(image, size=(self.batch_size, 320, 320, 3))
        return cropped, target

    def center_crop(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Resizes a batch of images to (self.resize_pre_crop, self.resize_pre_crop) and
        then takes central crop of (self.crop_size, self.crop_size)

        Args:
            image: Batch of images to perform center crop on.
            target: Target tensor.

        Returns:
            Center cropped example with batch of images and targets with same dimensions.
        """
        aug_images = tf.image.resize(
            image, (self.resize_pre_crop, self.resize_pre_crop)
        )
        aug_images = tf.image.central_crop(
            aug_images, float(self.crop_size) / float(self.resize_pre_crop)
        )
        return aug_images, target

    def validation_crop(self, example: dict):
        img = example["image"]
        h = example["height"]
        w = example["width"]

        if w < h and w != self.resize_pre_crop:
            w_resize, h_resize = tf.cast(self.resize_pre_crop, tf.int32), tf.cast(
                ((h / w) * self.resize_pre_crop), tf.int32
            )
            img = tf.image.resize(img, (h_resize, w_resize))
        elif h <= w and h != self.resize_pre_crop:
            w_resize, h_resize = tf.cast(
                ((w / h) * self.resize_pre_crop), tf.int32
            ), tf.cast(self.resize_pre_crop, tf.int32)
            img = tf.image.resize(img, (h_resize, w_resize))
        else:
            w_resize = tf.cast(w, tf.int32)
            h_resize = tf.cast(h, tf.int32)
            img = tf.image.resize(img, (h_resize, w_resize))

        x = tf.cast(tf.math.ceil((w_resize - self.crop_size) / 2), tf.int32)
        y = tf.cast(tf.math.ceil((h_resize - self.crop_size) / 2), tf.int32)

        if x <= 0 or y <= 0:
            tf.print(x, y)
        img = img[y : (y + self.crop_size), x : (x + self.crop_size), :]
        img = tf.cast(
            tf.math.round(tf.image.resize(img, (self.crop_size, self.crop_size))),
            tf.uint8,
        )

        return {
            "image": tf.cast(img, tf.uint8),
            "height": self.crop_size,
            "width": self.crop_size,
            "filename": example["filename"],
            "label": example["label"],
            "synset": example["synset"],
        }

    def _one_hot_encode_example(self, example: dict) -> tuple:
        """Takes an example having keys 'image' and 'label' and returns example
        with keys 'image' and 'target'. 'target' is one hot encoded.

        Args:
            example: an example dict having keys 'image' and 'label'.

        Returns:
            Tuple having structure (image_tensor, targets_tensor).
        """
        return (example["image"], tf.one_hot(example["label"], self.num_classes))

    def _mixup(self, image, label) -> Tuple:
        """
        Function to apply mixup augmentation. To be applied after
        one hot encoding and before batching.

        Args:
            entry1: Entry from first dataset. Should be one hot encoded and batched.
            entry2: Entry from second dataset. Must be one hot encoded and batched.

        Returns:
            Tuple with same structure as the entries.
        """
        image1, label1 = image, label
        image2, label2 = tf.reverse(image, axis=[0]), tf.reverse(label, axis=[0])

        image1 = tf.cast(image1, tf.float32)
        image2 = tf.cast(image2, tf.float32)

        alpha = [self.mixup_alpha]
        dist = tfd.Beta(alpha, alpha)
        l = dist.sample(1)[0][0]

        img = l * image1 + (1 - l) * image2
        lab = l * label1 + (1 - l) * label2

        img = tf.cast(
            tf.math.round(tf.image.resize(img, (self.crop_size, self.crop_size))),
            tf.uint8,
        )

        return img, lab

    def _get_random_dims(self, area, height, width, max_iter=10):
        """
        Working logic:
        1. Initialize values, start for loop and generate required random values.
        2. If h_crop and w_crop (i.e. the generated values) are lesser than image dimensions,
            then generate random x and y values.
        3. Cache these x and y values if they are useful (BOTH MUST BE USEFUL)
        4. If cached values exist, return them, else return bad values (atleast one of x and y will contain -1).

        Run cases:
        1. Cached values can be filled multiple times. Any time, they will be useful.
        2. If there were no (or partial) cached values after 10 iterations, we can safely apply validation crop

        Pros:
        1. The graph is constant, since we are not using break statement.
        2. From augmentation POV, the function remains constant.

        Cons:
        1. We run the function 10 times. However note that even if we encounter multiple valid values,
            all of them are valid. Thus this maintains corretness of the function.
        """
        w_crop = tf.cast(-1, tf.int32)
        h_crop = tf.cast(-1, tf.int32)

        x_cache = -1
        y_cache = -1

        for _ in tf.range(max_iter):
            target_area = (
                tf.random.uniform((), minval=self.area_factor, maxval=1) * area
            )
            aspect_ratio = tf.random.uniform((), minval=3.0 / 4.0, maxval=4.0 / 3.0)

            w_crop = tf.cast(
                tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32
            )
            h_crop = tf.cast(
                tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32
            )

            prob = tf.random.uniform((), minval=0.0, maxval=1.0)

            w_crop, h_crop = tf.cond(
                tf.math.greater(prob, tf.constant(0.5)),
                lambda: (h_crop, w_crop),
                lambda: (w_crop, h_crop),
            )
            x = -1
            y = -1

            if h_crop < height:
                y = tf.random.uniform(
                    (), minval=0, maxval=height - h_crop, dtype=tf.int32
                )

                if w_crop < width:
                    x = tf.random.uniform(
                        (), minval=0, maxval=width - w_crop, dtype=tf.int32
                    )
                    x_cache = x
                    y_cache = y

        if x_cache > -1:
            if y_cache > -1:
                return x_cache, y_cache, w_crop, h_crop
            else:
                return x, y, w_crop, h_crop
        else:
            return x, y, w_crop, h_crop

    def _inception_style_crop_single(self, example, max_iter=10):
        """
        Working logic:
        1. Get random values from generate random dims function (see its docstring).
        2. If the values are good (both > -1) then do inception style cropping
        2. In all other cases do valiation cropping


        """

        height = tf.cast(example["height"], tf.int32)
        width = tf.cast(example["width"], tf.int32)
        area = tf.cast(height * width, tf.float32)

        x, y, w_crop, h_crop = self._get_random_dims(area, height, width)

        img = tf.cast(example["image"], tf.uint8)
        w = width
        h = height

        if x > -1:
            if y > -1:
                # Inception
                w_resize = tf.cast(0, tf.int32)
                h_resize = tf.cast(0, tf.int32)
                img = img[y : y + h_crop, x : x + w_crop, :]
                img = tf.cast(
                    tf.math.round(
                        tf.image.resize(img, (self.crop_size, self.crop_size))
                    ),
                    tf.uint8,
                )
            else:
                # Validation
                if w < h:
                    w_resize, h_resize = tf.cast(
                        self.resize_pre_crop, tf.int32
                    ), tf.cast(((h / w) * self.resize_pre_crop), tf.int32)
                    img = tf.image.resize(img, (h_resize, w_resize))
                elif h <= w:
                    w_resize, h_resize = tf.cast(
                        ((w / h) * self.resize_pre_crop), tf.int32
                    ), tf.cast(self.resize_pre_crop, tf.int32)
                    img = tf.image.resize(img, (h_resize, w_resize))
                else:
                    w_resize = tf.cast(w, tf.int32)
                    h_resize = tf.cast(h, tf.int32)
                    img = tf.image.resize(img, (h_resize, w_resize))

                x = tf.cast(tf.math.ceil((w_resize - self.crop_size) / 2), tf.int32)
                y = tf.cast(tf.math.ceil((h_resize - self.crop_size) / 2), tf.int32)

                img = img[y : (y + self.crop_size), x : (x + self.crop_size), :]
                img = tf.cast(
                    tf.math.round(
                        tf.image.resize(img, (self.crop_size, self.crop_size))
                    ),
                    tf.uint8,
                )

        else:
            # Valiadation
            if w < h:
                w_resize, h_resize = tf.cast(self.resize_pre_crop, tf.int32), tf.cast(
                    ((h / w) * self.resize_pre_crop), tf.int32
                )
                img = tf.image.resize(img, (h_resize, w_resize))
            elif h <= w:
                w_resize, h_resize = tf.cast(
                    ((w / h) * self.resize_pre_crop), tf.int32
                ), tf.cast(self.resize_pre_crop, tf.int32)
                img = tf.image.resize(img, (h_resize, w_resize))
            else:
                w_resize = tf.cast(w, tf.int32)
                h_resize = tf.cast(h, tf.int32)
                img = tf.image.resize(img, (h_resize, w_resize))

            x = tf.cast(tf.math.ceil((w_resize - self.crop_size) / 2), tf.int32)
            y = tf.cast(tf.math.ceil((h_resize - self.crop_size) / 2), tf.int32)

            img = img[y : (y + self.crop_size), x : (x + self.crop_size), :]
            img = tf.cast(
                tf.math.round(tf.image.resize(img, (self.crop_size, self.crop_size))),
                tf.uint8,
            )

        return {
            "image": img,
            "height": self.crop_size,
            "width": self.crop_size,
            "filename": example["filename"],
            "label": example["label"],
            "synset": example["synset"],
        }

    def make_dataset(self):

        ds = self._read_tfrecs()

        if self.no_aug:
            ds = ds.map(
                lambda image, label: (tf.cast(image, tf.uint8), label),
                num_parallel_calls=AUTO,
            )
            return ds

        if self.default_augment:
            ds = ds.map(self._inception_style_crop_single, num_parallel_calls=AUTO)
            ds = ds.prefetch(AUTO)
            ds = ds.map(self._one_hot_encode_example, num_parallel_calls=AUTO)
            ds = ds.map(self.random_flip, num_parallel_calls=AUTO)

            if self.color_jitter:
                ds = ds.map(self._color_jitter, num_parallel_calls=AUTO)
            ds = ds.repeat()
            ds = ds.batch(self.batch_size, drop_remainder=False)
            if self.mixup:
                ds = ds.map(self._mixup, num_parallel_calls=AUTO)
            ds = ds.map(self._pca_jitter, num_parallel_calls=AUTO)

            # ds = ds.map(self._inception_style_crop, num_parallel_calls=AUTO)

        elif self.val_augment:
            ds = ds.map(self.validation_crop, num_parallel_calls=AUTO)
            #             ds = ds.prefetch(AUTO)
            ds = ds.map(self._one_hot_encode_example, num_parallel_calls=AUTO)
            ds = ds.repeat()
            ds = ds.batch(self.batch_size, drop_remainder=False)

        else:
            ds = ds.map(self.augment_fn, num_parallel_calls=AUTO)
        ds = ds.prefetch(AUTO)

        return ds
