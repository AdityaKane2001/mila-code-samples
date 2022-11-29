"""Contains utility functions for training."""

import logging
import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, List, Type, Union

import tensorflow as tf
import tensorflow_addons as tfa
import yaml
from wandb.keras import WandbCallback

PI = math.pi

logging.basicConfig(
    format="%(asctime)s %(levelname)s : %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


@dataclass
class PreprocessingConfig:
    tfrecs_filepath: List[str]
    batch_size: int
    image_size: int
    area_factor: float
    crop_size: int
    resize_pre_crop: int
    augment_fn: Union[str, Callable]
    num_classes: int
    color_jitter: bool
    mixup: bool
    mixup_alpha: float


@dataclass
class RegNetYConfig:
    """
    Dataclass for architecture configuration for RegNetY.

    Args:
        name: Name of the model eg. "RegNetY200MF"
        flops: Flops of the model eg. "400MF" (Processing one image requires
            400 million floating point operations (multiplication, addition))
        depths: List of depths for every stage
        widths: List of  widths (number of channels) after every stage
        group_width: Integer denoting groups in every convolution layer
        bottleneck_ratio: Integer specifying bottleneck ratio
        se_ratio: Float denoting squeeze and excite ratio
        wa: Integer, slope used in linear parameterization
        w0: Integer, inital value used in linear parameterization
        wm: Float, quantization parameter
    """

    name: str
    flops: str
    num_classes: int
    depths: List[int]
    widths: List[int]
    group_width: int
    bottleneck_ratio: int
    se_ratio: float
    wa: int
    w0: int
    wm: float


@dataclass
class TrainConfig:
    """
    Dataclass of training configuration for RegNetY

    Args:
        optimizer: One of "sgd", "adam", "adamw"
        base_lr: Base learning rate for training
        warmup_epochs: Number of epochs used for warmup
        warmup_factor: Gradual linear warmup factor
        total_epochs: Number of training epochs
        weight_decay: Weight decay to be used in optimizer
        momentum: Momentum to be used in optimizer
        lr_schedule: One of "constant" or "half_cos"
        log_dir: Path to store logs
        model_dir: Path to store model checkpoints
    """

    optimizer: str
    base_lr: float
    warmup_epochs: int
    warmup_factor: float
    total_epochs: int
    weight_decay: float
    momentum: float
    label_smoothing: float
    lr_schedule: str
    log_dir: str
    model_dir: str


def get_preprocessing_config(
    tfrecs_filepath: List[str] = None,
    batch_size: int = 1024,
    image_size: int = 512,
    area_factor: float = 0.08,
    crop_size: int = 224,
    resize_pre_crop: int = 320,
    augment_fn: Union[str, Callable] = "default",
    num_classes: int = 1000,
    color_jitter: bool = False,
    mixup: bool = True,
    mixup_alpha: float = 0.2,
):

    return PreprocessingConfig(
        tfrecs_filepath=tfrecs_filepath,
        batch_size=batch_size,
        image_size=image_size,
        area_factor=area_factor,
        crop_size=crop_size,
        resize_pre_crop=resize_pre_crop,
        augment_fn=augment_fn,
        num_classes=num_classes,
        color_jitter=color_jitter,
        mixup=mixup,
        mixup_alpha=mixup_alpha,
    )


def get_train_config(
    optimizer: str = "adamw",
    base_lr: float = 0.001 * 8,
    warmup_epochs: int = 5,
    warmup_factor: float = 0.1,
    total_epochs: int = 100,
    weight_decay: float = 5e-5,
    momentum: float = 0.9,
    label_smoothing: float = 0.0,
    lr_schedule: str = "half_cos",
    log_dir: str = "",
    model_dir: str = "",
):
    return TrainConfig(
        optimizer=optimizer,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        warmup_factor=warmup_factor,
        total_epochs=total_epochs,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        momentum=momentum,
        lr_schedule=lr_schedule,
        log_dir=log_dir,
        model_dir=model_dir,
    )


def get_optimizer(cfg):
    if cfg.optimizer == "sgd":
        opt = tfa.optimizers.SGDW(
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.base_lr,
            momentum=cfg.momentum,
            nesterov=True,
        )

        #         opt = tfa.optimizers.MovingAverage(
        #             opt,
        #             average_decay=0.99999,
        #             start_step=6250,
        #         )
        return opt
    elif cfg.optimizer == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=cfg.base_lr,
        )

    elif cfg.optimizer == "adamw":
        return tfa.optimizers.AdamW(
            weight_decay=cfg.weight_decay, learning_rate=cfg.base_lr
        )

    elif cfg.optimizer == "nadamw":
        opt = tfa.optimizers.extend_with_decoupled_weight_decay(
            tf.keras.optimizers.Nadam
        )
        return opt(weight_decay=cfg.weight_decay, learning_rate=cfg.base_lr)

    else:
        raise NotImplementedError(f"Optimizer choice not supported: {cfg.optimizer}")


def get_train_schedule(cfg):
    if cfg.lr_schedule == "half_cos":

        def half_cos_schedule(epoch, lr):
            # Taken from pycls/pycls/core/optimizer.py, since not clear from paper.
            if epoch < cfg.warmup_epochs:
                new_lr = (
                    0.5
                    * (1.0 + tf.math.cos(PI * epoch / cfg.total_epochs))
                    * cfg.base_lr
                )
                alpha = epoch / cfg.warmup_epochs
                warmup_factor = cfg.warmup_factor * (1.0 - alpha) + alpha
                return new_lr * warmup_factor
            else:
                new_lr = (
                    0.5
                    * (1.0 + tf.math.cos(PI * epoch / cfg.total_epochs))
                    * cfg.base_lr
                )
                return new_lr

        return half_cos_schedule

    elif cfg.lr_schedule == "constant":
        return cfg.base_lr


def get_callbacks(cfg, timestr):
    lr_callback = tf.keras.callbacks.LearningRateScheduler(get_train_schedule(cfg))
    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(cfg.log_dir, timestr), histogram_freq=1
    )  # profile_batch="0,1023"

    best_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            cfg.model_dir,
            timestr,
            "best_model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}",
        ),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )
    all_models_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            cfg.model_dir,
            timestr,
            "all_model_epoch_{epoch:02d}",
        ),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False,
    )

    average_saving_callback = tfa.callbacks.AverageModelCheckpoint(
        update_weights=False,
        filepath=os.path.join(
            cfg.model_dir,
            timestr,
            "average_model_epoch_{epoch:02d}",
        ),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False,
    )

    return [
        lr_callback,
        #         tboard_callback,
        # best_model_checkpoint_callback,
        #         average_saving_callback,
        all_models_checkpoint_callback,
        WandbCallback(),
    ]


def get_config_dict(train_prep_cfg, val_prep_cfg, train_cfg, misc=None):
    config_dict = dict()
    train_prep_dict = asdict(train_prep_cfg)
    val_prep_dict = asdict(val_prep_cfg)
    del train_prep_dict["tfrecs_filepath"]
    del val_prep_dict["tfrecs_filepath"]
    config_dict["train_prep"] = train_prep_dict
    config_dict["val_prep"] = val_prep_dict
    config_dict["train_cfg"] = asdict(train_cfg)
    config_dict["misc"] = misc
    return config_dict


# def make_model(flops, train_cfg):
#     optim = get_optimizer(train_cfg)
#     model = regnety.models.model.RegNetY(flops)
#     model.compile(
#         loss=tf.keras.losses.CategoricalCrossentropy(
#             from_logits=True, label_smoothing=0.2),
#         optimizer=optim,
#         metrics=[
#             tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
#             tf.keras.metrics.TopKCategoricalAccuracy(
#                 5, name="top-5-accuracy"),
#         ],
#     )

#     return model


def connect_to_tpu(tpu_address: str = None):
    if tpu_address is not None:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address
        )
        if tpu_address not in ("", "local"):
            tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        logging.info(f"Running on TPU {cluster_resolver.master()}")
        logging.info(f"REPLICAS: {strategy.num_replicas_in_sync}")
        return cluster_resolver, strategy
    else:
        try:
            cluster_resolver = (
                tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            )
            strategy = tf.distribute.TPUStrategy(cluster_resolver)
            logging.info(f"Running on TPU {cluster_resolver.master()}")
            logging.info(f"REPLICAS: {strategy.num_replicas_in_sync}")
            return cluster_resolver, strategy
        except:
            logging.warning("No TPU detected.")
            mirrored_strategy = tf.distribute.MirroredStrategy()
            return None, mirrored_strategy
