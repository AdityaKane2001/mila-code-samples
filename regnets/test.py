"""Script for training RegNetY. Supports TPU training."""

import argparse
import json
import logging
import os
from datetime import datetime

import tensorflow as tf
import wandb

from dataset import ImageNet
from utils import *
from wandb.keras import WandbCallback

# Contrived example of generating a module named as a string


NORMALIZED = False


log_location = "gs://ak-us-train"
train_tfrecs_filepath = tf.io.gfile.glob("gs://ak-imagenet-new/train/train_*.tfrecord")
val_tfrecs_filepath = tf.io.gfile.glob("gs://ak-imagenet-new/valid/valid_*.tfrecord")

logging.basicConfig(
    format="%(asctime)s %(levelname)s : %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

cluster_resolver, strategy = connect_to_tpu()

train_cfg = get_train_config(
    optimizer="sgd",
    base_lr=0.1 * strategy.num_replicas_in_sync,
    warmup_epochs=5,
    warmup_factor=0.1,
    total_epochs=100,
    weight_decay=5e-5,
    momentum=0.9,
    label_smoothing=0.0,
    lr_schedule="half_cos",
    log_dir=log_location + "/logs",
    model_dir=log_location + "/models",
)


train_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=train_tfrecs_filepath,
    batch_size=1024,
    image_size=512,
    crop_size=224,
    resize_pre_crop=256,
    augment_fn="default",
    num_classes=1000,
    color_jitter=False,
    mixup=False,
)

val_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=val_tfrecs_filepath,
    batch_size=1024,
    image_size=512,
    crop_size=224,
    resize_pre_crop=256,
    augment_fn="val",
    num_classes=1000,
    color_jitter=False,
    mixup=False,
)

misc_dict = {"Rescaling": "1/255", "Normalization": "None"}

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%Hh%Mm")

config_dict = get_config_dict(train_prep_cfg, val_prep_cfg, train_cfg, misc=misc_dict)

# logging.info(config_dict)

wandb.init(
    entity="compyle",
    project="keras-regnet-training",
    job_type="train",
    name="regnetx002" + "_" + date_time,
    config=config_dict,
)
train_cfg = wandb.config.train_cfg
