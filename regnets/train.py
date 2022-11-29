"""Script for training RegNets. Supports TPU training."""

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
train_tfrecs_filepath = tf.io.gfile.glob(
    "gs://ak-imagenet-new/train-2/train_*.tfrecord"
)
val_tfrecs_filepath = tf.io.gfile.glob("gs://ak-imagenet-new/valid-2/valid_*.tfrecord")

logging.basicConfig(
    format="%(asctime)s %(levelname)s : %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

cluster_resolver, strategy = connect_to_tpu()


train_cfg = get_train_config(
    optimizer="adamw",
    base_lr=0.001
    * strategy.num_replicas_in_sync,  #################################################change this!!
    warmup_epochs=5,
    warmup_factor=0.1,
    total_epochs=100,
    weight_decay=1e-5,
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
    area_factor=0.16,
    crop_size=224,
    resize_pre_crop=256,
    augment_fn="default",
    num_classes=1000,
    color_jitter=False,
    mixup=True,
    mixup_alpha=0.2,
)

val_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=val_tfrecs_filepath,
    batch_size=1024,
    image_size=512,
    area_factor=0.25,
    crop_size=224,
    resize_pre_crop=256,
    augment_fn="val",
    num_classes=1000,
    color_jitter=False,
    mixup=False,
    mixup_alpha=0.0,
)

misc_dict = {
    "Rescaling": "1/255",
    "Normalization": "None",
}

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%Hh%Mm%Ss")

config_dict = get_config_dict(train_prep_cfg, val_prep_cfg, train_cfg, misc=misc_dict)

logging.info(config_dict)

wandb.init(
    entity="compyle",
    project="keras-regnet-training",
    job_type="train",
    name="regnety004" + "_" + date_time,
    config=config_dict,
)
# train_cfg = wandb.config.train_cfg
# train_cfg = from_dict(data_class=TrainConfig, data=train_cfg)
logging.info(f"Training options detected: {train_cfg}")
logging.info("Preprocessing options detected.")
logging.info(
    f"Training on TFRecords: {train_prep_cfg.tfrecs_filepath[0]} to {train_prep_cfg.tfrecs_filepath[-1]}"
)
logging.info(
    f"Validating on TFRecords: {val_prep_cfg.tfrecs_filepath[0]} to {val_prep_cfg.tfrecs_filepath[-1]}"
)

INIT_EPOCH = 80

with strategy.scope():
    optim = get_optimizer(train_cfg)

    model = tf.keras.applications.RegNetY004()
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=train_cfg.label_smoothing
        ),
        optimizer=optim,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    if INIT_EPOCH > 0:
        model.load_weights(
            "gs://ak-us-train/models/12_09_2021_05h22m18s/all_model_epoch_"
            + f"{INIT_EPOCH:02}"
        )
    logging.info("Model loaded")

train_ds = ImageNet(train_prep_cfg).make_dataset()
# train_ds = train_ds.shuffle(300)
val_ds = ImageNet(val_prep_cfg).make_dataset()
# val_ds = val_ds.shuffle(48)


callbacks = get_callbacks(train_cfg, date_time)


if INIT_EPOCH > 0:
    count = 1252 * (INIT_EPOCH - 3)

    for i in range(len(callbacks)):
        try:
            callbacks[i].count = count
        except:
            pass

history = model.fit(
    train_ds,
    epochs=train_cfg.total_epochs,
    steps_per_epoch=1252,
    validation_data=val_ds,
    callbacks=callbacks,
    validation_steps=49,
    initial_epoch=INIT_EPOCH - 3,
)

with tf.io.gfile.GFile(
    os.path.join(train_cfg.log_dir, "history_%s.json" % date_time), "a+"
) as f:
    json.dump(str(history.history), f)
