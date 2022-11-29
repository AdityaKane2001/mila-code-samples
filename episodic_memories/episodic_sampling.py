import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import json
import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
import random
import wandb
from copy import deepcopy

from ml_collections import ConfigDict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

cfg = ConfigDict()
cfg.test_split = 0.2
cfg.batch_size = 256
cfg.task_0_lr = 1e-4
cfg.task_1_lr = 5e-6
cfg.residual_mlp_dropout_rate = 0.2
cfg.task_0_weight_decay = 1e-5
cfg.task_1_weight_decay = 2e-5
cfg.samples_per_class = 25
cfg.hidden_dims = 1024
cfg.op = "mul"
cfg.epochs = 25
cfg.task_order = "yesno imagecondition roadcondition"

cfg.half_batch_size = cfg.batch_size // 2


class VQADataset(Dataset):
    def __init__(self, qdict, label_mapping=None):
        self.qdict = qdict
        self.label_mapping = label_mapping
        self.reset_index()

    def reset_index(self):
        new_qdict = dict()
        for idx, value in enumerate(self.qdict.values()):
            new_qdict[idx] = value
        self.qdict = new_qdict

    def __len__(self):
        return len(self.qdict.keys())

    def __getitem__(self, idx):
        # "0":{"Image_ID":"10165.JPG","Question":"What is the overall condition of the given image?","Ground_Truth":"flooded","Question_Type":"Condition_Recognition"}
        row = self.qdict[idx]
        img_feat = all_images_features[row["Image_ID"]]
        q_feat = all_questions_features[row["Question"]]

        return (
            img_feat.float(),
            q_feat.squeeze().float(),
            self.label_mapping.index(str(row["Ground_Truth"])),
        )


############## Experience replay logic


class ExperienceReplay:
    def __init__(self, samples_per_class=10, num_classes=20, half_batch_size=8):
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.half_batch_size = half_batch_size

        self.memory_size = self.samples_per_class * self.num_classes

        self.memory = []

    def update_memory(self, current_batch, elapsed_examples=0):
        list_of_examples = unbatch(current_batch)

        counter = 0

        for example in list_of_examples:
            if len(self.memory) < self.memory_size:
                self.memory.append(example)
            else:
                idx = random.randint(0, elapsed_examples + counter)
                if idx < self.memory_size:
                    self.memory[idx] = example

            counter += 1
        return None

    def get_from_memory(self, num_samples):
        return random.choices(self.memory, k=num_samples)


############## Utilities for Experience replay


def unbatch(half_batch):
    """
    Unbatches a batch into list of examples.

    Args:
        batch: A batch of examples with the structure :
        [torch.Tensor, torch.Tensor, torch.Tensor]

    Returns:
        list of unbatched examples: [[torch.Tensor, torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor, torch.Tensor]]

    """
    list_of_examples = []

    num_examples = len(half_batch[0])

    for idx in range(num_examples):
        list_of_examples.append(
            [half_batch[0][idx], half_batch[1][idx], half_batch[2][idx]]
        )

    return list_of_examples


def batch(list_of_examples):
    """
    Batches unbatched examples into one

    Args:
        list_of_examples: list of unbatched examples: [[torch.Tensor, torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor, torch.Tensor]]

    Returns:
        A batch of examples with the structure :
        [torch.Tensor, torch.Tensor, torch.Tensor]
    """
    img_feats = []
    q_feats = []
    labels = []
    for example in list_of_examples:
        img_feats.append(example[0])
        q_feats.append(example[1])
        labels.append(example[2])

    return torch.concat(img_feats), torch.concat(q_feats), torch.concat(labels)


def hello():
    print("hi")


def combine_batch_and_list(half_batch, list_of_examples):
    for example in list_of_examples:
        half_batch[0] = torch.concat([half_batch[0], example[0].unsqueeze(0)], dim=0)
        half_batch[1] = torch.concat([half_batch[1], example[1].unsqueeze(0)], dim=0)
        half_batch[2] = torch.concat([half_batch[2], example[2].unsqueeze(0)], dim=0)
    return half_batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def accuracy(pred, true):
    acc = np.sum((true == pred.argmax(-1)).astype(np.float32))
    return float(100 * acc / len(true))


def get_train_val_splits(jsondict):
    total = len(jsondict.keys())
    train, val = train_test_split(list(jsondict.keys()), test_size=cfg.test_split)

    print(len(train), len(val))

    train_dict = dict()
    val_dict = dict()
    for i in train:
        train_dict[str(i)] = jsondict[str(i)]

    for j in val:
        val_dict[str(j)] = jsondict[str(j)]

    return train_dict, val_dict


def get_uniq_image_ids(jsondict):
    uniq_images = []
    for key, example in jsondict.items():
        if example["Image_ID"] not in uniq_images:
            uniq_images.append(example["Image_ID"])
    return uniq_images


def get_questions_for_imageid(jsondict, imageid):
    qs = dict()
    for key, example in jsondict.items():
        if example["Image_ID"] == imageid:
            qs[key] = example
    return qs


def get_train_val_splits_imagewise(jsondict):
    train_dict = dict()
    val_dict = dict()

    uniq_images = get_uniq_image_ids(jsondict)

    train, val = train_test_split(uniq_images, test_size=0.2)

    for imageid in train:
        train_dict.update(get_questions_for_imageid(jsondict, imageid))

    for imageid in val:
        val_dict.update(get_questions_for_imageid(jsondict, imageid))

    return train_dict, val_dict


def get_typewise_train_val_splits(train_dict, val_dict):
    train_road_condition = dict()
    train_yes_no = dict()
    train_image_condition = dict()

    counter = 0
    for example in train_dict.values():
        if example["Question_Type"] == "Yes_No":
            train_yes_no[str(counter)] = example
            counter += 1
        elif "road" in example["Question"]:
            train_road_condition[str(counter)] = example
            counter += 1
        elif "overall" in example["Question"]:
            train_image_condition[str(counter)] = example
            counter += 1

    val_road_condition = dict()
    val_yes_no = dict()
    val_image_condition = dict()

    counter = 0
    for example in val_dict.values():
        if example["Question_Type"] == "Yes_No":
            val_yes_no[str(counter)] = example
            counter += 1
        elif "road" in example["Question"]:
            val_road_condition[str(counter)] = example
            counter += 1
        elif "overall" in example["Question"]:
            val_image_condition[str(counter)] = example
            counter += 1

    return [train_image_condition, train_road_condition, train_yes_no], [
        val_image_condition,
        val_road_condition,
        val_yes_no,
    ]


def accuracy(pred, true):
    acc = np.sum((true == pred.argmax(-1)).astype(np.float32))
    return float(100 * acc / len(true))


label_mapping = [
    "flooded",  # condition
    "non flooded",  # condition
    "flooded,non flooded",  # condition
    "Yes",  # yes/no
    "No",  # yes/no
] + list(
    map(str, range(0, 51))
)  # counting

ORIGINAL_DATA_PATH = "/content/drive/MyDrive/floodnet_data/"
RESNET_FEATURE_PATH = (
    "/content/drive/MyDrive/continual_floodnet_data/floodnet_resnet_features/"
)
CONVNEXT_FEATURE_PATH = (
    "/content/drive/MyDrive/continual_floodnet_data/floodnet_convnext_features/"
)
CLIP_FEATURE_PATH = (
    "/content/drive/MyDrive/continual_floodnet_data/floodnet_clip_features/"
)

MODE = "CLIP"  # can be RESNET or CONVNEXT
if MODE == "RESNET":
    FEATURE_PATH = RESNET_FEATURE_PATH
    IMG_FEAT_DIMS = 2048
elif MODE == "CONVNEXT":
    FEATURE_PATH = CONVNEXT_FEATURE_PATH
    IMG_FEAT_DIMS = 1536
elif MODE == "CLIP":
    FEATURE_PATH = CLIP_FEATURE_PATH
    IMG_FEAT_DIMS = 768
else:
    raise ValueError(f"Mode must be one of `RESNET` or `CONVNEXT`, got {MODE}")

all_images_features = dict()
for imagename in os.listdir(os.path.join(FEATURE_PATH, "Images/Train_Image")):
    all_images_features[imagename.replace(".pt", ".JPG")] = torch.load(
        os.path.join(os.path.join(FEATURE_PATH, "Images/Train_Image"), imagename)
    )
# 1234.JPG : torch.Tensor

all_questions_features = dict()
for imagename in os.listdir(os.path.join(FEATURE_PATH, "Questions")):
    all_questions_features[imagename.replace(".pt", "").replace("_", "?")] = torch.load(
        os.path.join(os.path.join(FEATURE_PATH, "Questions"), imagename)
    )
# "what is the overall condition of this image?": torch.Tensor


############## Model architecture


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dims, num_blocks=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dims = hidden_dims
        self.resblocks = list()
        self.resblocks = nn.Sequential(
            *[
                Residual(
                    nn.Sequential(
                        nn.Linear(self.hidden_dims, 512),
                        nn.Dropout(p=cfg.residual_mlp_dropout_rate),
                        nn.Linear(512, 256),
                        nn.Linear(256, 512),
                        nn.Dropout(p=cfg.residual_mlp_dropout_rate),
                        nn.Linear(512, self.hidden_dims),
                    )
                )
                for block_idx in range(self.num_blocks)
            ]
        )

    def forward(self, inputs):
        return self.resblocks(inputs)


class VQAModel(nn.Module):
    def __init__(
        self, op="cat", out_classes=5
    ):  # `op` can be one of `cat`, `add`, `mul`
        super().__init__()
        self.op = op
        if self.op == "add" or "mul":
            self.image = nn.Sequential(
                nn.Linear(in_features=IMG_FEAT_DIMS, out_features=1024),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=1024, out_features=512),
            )
            self.txt = nn.Linear(in_features=1024, out_features=512)
            self.linstack1 = nn.Sequential(
                nn.Linear(in_features=512, out_features=256),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=256, out_features=128),
            )
            self.linstack2 = deepcopy(self.linstack1)
            self.linstack3 = deepcopy(self.linstack1)
            self.cls = nn.Linear(in_features=128 * 3, out_features=out_classes)

        elif self.op == "cat":
            self.image = nn.Identity()
            self.text = nn.Identity()
            self.linstack1 = nn.Sequential(
                nn.Linear(
                    in_features=2560, out_features=1024
                ),  # 2560 = 1024 (text) + 1536 (image)
                nn.Dropout(p=0.2),
                nn.Linear(in_features=1024, out_features=512),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=512, out_features=256),
                nn.Linear(in_features=256, out_features=128),
            )
            self.linstack2 = deepcopy(self.linstack1)
            self.linstack3 = deepcopy(self.linstack1)
            self.cls = nn.Linear(in_features=128 * 3, out_features=out_classes)
        else:
            raise ValueError(f"`op` must be one of `cat`, `add`, `mul`. Got {op}")

    def forward(self, batch):
        if self.op == "cat":
            x = torch.concat([batch[0], batch[1]], dim=-1)
            x1 = self.linstack1(x)
            x2 = self.linstack2(x)
            x3 = self.linstack3(x)
            x = torch.concat([x1, x2, x3], dim=-1)
            return self.cls(x)
        elif self.op == "add":
            img = self.image(batch[0])
            txt = self.txt(batch[1])
            x = img + txt
            # vec = torch.dot(img, txt) # check if they are 1-d tensors
            x1 = self.linstack1(x)
            x2 = self.linstack2(x)
            x3 = self.linstack3(x)
            x = torch.concat([x1, x2, x3], dim=-1)

            return self.cls(x)
        elif self.op == "mul":
            img = self.image(batch[0])
            txt = self.txt(batch[1])
            x = img * txt
            # vec = torch.dot(img, txt) # check if they are 1-d tensors
            x1 = self.linstack1(x)
            x2 = self.linstack2(x)
            x3 = self.linstack3(x)
            x = torch.concat([x1, x2, x3], dim=-1)
            return self.cls(x)


class VQAResidualMLPModel(nn.Module):
    def __init__(self, op="cat", out_classes=5, hidden_dims=512, device=None):
        super().__init__()
        self.op = op
        if self.op == "add" or "mul":
            self.image = nn.Sequential(
                nn.Linear(in_features=IMG_FEAT_DIMS, out_features=1024),  ###########
                nn.Dropout(p=0.2),
                nn.Linear(in_features=1024, out_features=hidden_dims),
            )
            self.txt = nn.Linear(in_features=768, out_features=hidden_dims)  ###########

            self.linstack1 = ResidualMLPBlock(hidden_dims=hidden_dims)
            self.linstack2 = ResidualMLPBlock(hidden_dims=hidden_dims)
            self.linstack3 = ResidualMLPBlock(hidden_dims=hidden_dims)
            self.cls = nn.Linear(in_features=hidden_dims * 3, out_features=out_classes)

        elif self.op == "cat":
            self.image = nn.Identity()
            self.text = nn.Identity()
            self.linstack1 = nn.Sequential(
                nn.Linear(in_features=2560, out_features=hidden_dims),
                ResidualMLPBlock(hidden_dims=hidden_dims),
            )
            self.linstack2 = nn.Sequential(
                nn.Linear(in_features=2560, out_features=hidden_dims),
                ResidualMLPBlock(hidden_dims=hidden_dims),
            )
            self.linstack3 = nn.Sequential(
                nn.Linear(in_features=2560, out_features=hidden_dims),
                ResidualMLPBlock(hidden_dims=hidden_dims),
            )
            self.cls = nn.Linear(in_features=hidden_dims * 3, out_features=out_classes)
        else:
            raise ValueError(f"`op` must be one of `cat`, `add`, `mul`. Got {op}")
        self.linstack1 = self.linstack1.to(device)
        self.linstack2 = self.linstack2.to(device)
        self.linstack3 = self.linstack3.to(device)

    def forward(self, batch):

        if self.op == "cat":
            x = torch.concat([batch[0], batch[1]], dim=-1)
            x1 = self.linstack1(x)
            x2 = self.linstack2(x)
            x3 = self.linstack3(x)
            x = torch.concat([x1, x2, x3], dim=-1)
            return self.cls(x)
        elif self.op == "add":
            img = self.image(batch[0])
            txt = self.txt(batch[1])
            x = img + txt
            # vec = torch.dot(img, txt) # check if they are 1-d tensors
            x1 = self.linstack1(x)
            x2 = self.linstack2(x)
            x3 = self.linstack3(x)
            x = torch.concat([x1, x2, x3], dim=-1)

            return self.cls(x)
        elif self.op == "mul":
            img = self.image(batch[0])
            txt = self.txt(batch[1])
            x = img * txt
            # vec = torch.dot(img, txt) # check if they are 1-d tensors
            x1 = self.linstack1(x)
            x2 = self.linstack2(x)
            x3 = self.linstack3(x)
            x = torch.concat([x1, x2, x3], dim=-1)
            return self.cls(x)


for task_order in [
    "yesno imagecondition roadcondition",
    "yesno roadcondition imagecondition",
    "imagecondition yesno roadcondition",
    "imagecondition roadcondition yesno",
    "roadcondition yesno imagecondition",
    "roadcondition imagecondition yesno",
]:
    qdict = json.load(
        open(
            "/content/drive/MyDrive/floodnet_data/Questions/Training Question.json", "r"
        )
    )

    train_dict, val_dict = get_train_val_splits_imagewise(qdict)
    train_tasks, val_tasks = get_typewise_train_val_splits(train_dict, val_dict)

    cfg.task_order = task_order
    new_train_tasks = []
    new_val_tasks = []

    # "yes_no image_condition road_condition"
    # [train_image_condition, train_road_condition, train_yes_no]
    for task_name in cfg.task_order.split():
        if task_name == "yesno":
            new_train_tasks.append(train_tasks[2])
            new_val_tasks.append(val_tasks[2])
        elif task_name == "imagecondition":
            new_train_tasks.append(train_tasks[0])
            new_val_tasks.append(val_tasks[0])
        elif task_name == "roadcondition":
            new_train_tasks.append(train_tasks[1])
            new_val_tasks.append(val_tasks[1])

    train_tasks = new_train_tasks
    val_tasks = new_val_tasks

    val_preserve = deepcopy(val_tasks)

    # Make continual
    val_tasks[2].update(val_tasks[1])
    val_tasks[2].update(val_tasks[0])
    val_tasks[1].update(val_tasks[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make model
    model = VQAResidualMLPModel(
        op=cfg.op, hidden_dims=cfg.hidden_dims, out_classes=5, device=device
    )

    for child in model.children():
        child.to(device)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    exp_replay = ExperienceReplay(
        samples_per_class=cfg.samples_per_class,
        num_classes=5,
        half_batch_size=cfg.half_batch_size,
    )

    NUM_TASKS = 3
    counter = 0
    EPOCHS = cfg.epochs

    now = datetime.now()
    timestr = now.strftime("%d_%m_%Hh%Mm%Ss")

    run_name = "_".join(
        ["reservoir", cfg.task_order.replace(" ", "_"), timestr, model.op]
    )

    wandb.init(
        project="continual_vqa", entity="compyle", name=run_name, config=cfg.to_dict()
    )

    for tasknum in range(NUM_TASKS):

        os.makedirs(f"./ckpts/{timestr}", exist_ok=True)
        train_dl = DataLoader(
            VQADataset(train_tasks[tasknum]),
            batch_size=cfg.half_batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_dl = DataLoader(
            VQADataset(val_tasks[tasknum]),
            batch_size=cfg.half_batch_size,
            shuffle=True,
            num_workers=4,
        )

        # preserve individual task data for metrics
        previous_tasks_dl = []
        for prevtasknum in range(tasknum + 1):
            previous_tasks_dl.append(
                DataLoader(
                    VQADataset(val_preserve[prevtasknum]),
                    batch_size=cfg.half_batch_size,
                    shuffle=True,
                    num_workers=4,
                )
            )

        if tasknum == 0:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.task_0_lr,
                weight_decay=cfg.task_0_weight_decay,
            )
        elif tasknum == 1:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.task_1_lr,
                weight_decay=cfg.task_1_weight_decay,
            )

        for epoch in range(EPOCHS):
            print(f"########## Epoch {epoch}")
            model.train()
            optimizer.zero_grad()

            epoch_loss = []
            epoch_acc = []

            val_loss = []
            val_acc = []

            for batch in tqdm.tqdm(train_dl):
                preserved_batch = deepcopy(batch)
                if tasknum > 0:
                    batch = combine_batch_and_list(
                        batch, exp_replay.get_from_memory(cfg.half_batch_size)
                    )

                batch = [elem.to(device) for elem in batch]

                outputs = model(batch)
                loss = loss_fn(outputs, batch[2])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                acc = accuracy(
                    outputs.detach().cpu().numpy(), batch[2].detach().cpu().numpy()
                )

                epoch_loss.append(loss.detach().cpu().numpy())
                epoch_acc.append(acc)
                if epoch == EPOCHS - 1:
                    counter += len(batch[2])

            if epoch == EPOCHS - 1:
                exp_replay.update_memory(preserved_batch, elapsed_examples=counter)

            # Current task evaluation
            model.eval()
            with torch.no_grad():
                for batch in tqdm.tqdm(val_dl):
                    batch = [elem.to(device) for elem in batch]
                    outputs = model(batch)
                    loss = loss_fn(outputs, batch[2])

                    acc = accuracy(
                        outputs.detach().cpu().numpy(), batch[2].detach().cpu().numpy()
                    )

                    val_loss.append(loss.detach().cpu().numpy())
                    val_acc.append(acc)

            metrics_dict = {
                f"task_{tasknum}_train_loss": np.mean(epoch_loss),
                f"task_{tasknum}_val_loss": np.mean(val_loss),
                f"task_{tasknum}_train_acc": np.mean(epoch_acc),
                f"task_{tasknum}_val_acc": np.mean(val_acc),
            }

            print(f"Train loss: {np.mean(epoch_loss)}", end="\t")
            print(f"Val loss: {np.mean(val_loss)}", end="\t")
            print(f"Train acc: {np.mean(epoch_acc)}", end="\t")
            print(f"Val acc: {np.mean(val_acc)}")

            # Non current task evaluation
            model.eval()
            for prevtasknum in range(tasknum + 1):
                val_loss = []
                val_acc = []

                with torch.no_grad():
                    for batch in tqdm.tqdm(previous_tasks_dl[prevtasknum]):
                        batch = [elem.to(device) for elem in batch]
                        outputs = model(batch)
                        loss = loss_fn(outputs, batch[2])

                        acc = accuracy(
                            outputs.detach().cpu().numpy(),
                            batch[2].detach().cpu().numpy(),
                        )

                        val_loss.append(loss.detach().cpu().numpy())
                        val_acc.append(acc)

                metrics_dict[
                    f"task_{tasknum}_prevtask_{prevtasknum}_val_acc"
                ] = np.mean(val_acc)
                metrics_dict[
                    f"task_{tasknum}_prevtask_{prevtasknum}_val_loss"
                ] = np.mean(val_loss)

            wandb.log(metrics_dict)
    wandb.finish()
