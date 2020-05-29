import json

system_dict = {}


def initialize_train_dataloader(
    data_dir="data/VOCtrainval_11-May-2012",
    batch_size=8,
    base_size=400,
    crop_size=380,
    augment=True,
    shuffle=True,
    scale=True,
    flip=True,
    rotate=True,
    blur=False,
    split="train_aug",
    num_workers=8,
):
    system_dict["train_loader"] = {}
    system_dict["train_loader"]["type"] = "VOC"
    system_dict["train_loader"]["args"] = {}
    system_dict["train_loader"]["args"]["data_dir"] = data_dir
    system_dict["train_loader"]["args"]["batch_size"] = batch_size
    system_dict["train_loader"]["args"]["base_size"] = base_size
    system_dict["train_loader"]["args"]["crop_size"] = crop_size
    system_dict["train_loader"]["args"]["augment"] = augment
    system_dict["train_loader"]["args"]["shuffle"] = shuffle
    system_dict["train_loader"]["args"]["scale"] = scale
    system_dict["train_loader"]["args"]["flip"] = flip
    system_dict["train_loader"]["args"]["rotate"] = rotate
    system_dict["train_loader"]["args"]["blur"] = blur
    system_dict["train_loader"]["args"]["split"] = split
    system_dict["train_loader"]["args"]["num_workers"] = num_workers


def initialize_val_dataloader(
    data_dir="data/VOCtrainval_11-May-2012",
    batch_size=8,
    crop_size=380,
    split="val",
    num_workers=4,
):

    system_dict["val_loader"] = {}
    system_dict["val_loader"]["type"] = "VOC"
    system_dict["val_loader"]["args"] = {}
    system_dict["val_loader"]["args"]["data_dir"] = data_dir
    system_dict["val_loader"]["args"]["batch_size"] = batch_size
    system_dict["val_loader"]["args"]["crop_size"] = crop_size
    system_dict["val_loader"]["args"]["val"] = True
    system_dict["val_loader"]["args"]["split"] = split
    system_dict["val_loader"]["args"]["num_workers"] = num_workers


def initialize_model(
    model="PSPNet",
    backbone="resnet50",
    freeze_bn=False,
    freeze_backbone=False,
    use_synch_bn=True,
    n_gpu=1,
):
    system_dict["name"] = model
    system_dict["n_gpu"] = n_gpu
    system_dict["use_synch_bn"] = use_synch_bn
    system_dict["arch"] = {}
    system_dict["arch"]["type"] = model
    system_dict["arch"]["args"] = {}
    system_dict["arch"]["args"]["backbone"] = backbone
    system_dict["arch"]["args"]["freeze_bn"] = freeze_bn
    system_dict["arch"]["args"]["freeze_backbone"] = freeze_backbone


def initialize_optimizer_loss(
    optimizer="SGD",
    differential_lr=True,
    lr=0.01,
    weight_decay=1e-4,
    momentum=0.9,
    loss="CrossEntropyLoss2d",
    ignore_index=255,
    lr_scheduler_type="Poly",
):

    system_dict["optimizer"] = {}
    system_dict["optimizer"]["type"] = optimizer
    system_dict["optimizer"]["differential_lr"] = differential_lr
    system_dict["optimizer"]["args"] = {}
    system_dict["optimizer"]["args"]["lr"] = lr
    system_dict["optimizer"]["args"]["weight_decay"] = weight_decay
    system_dict["optimizer"]["args"]["momentum"] = momentum

    system_dict["loss"] = loss
    system_dict["ignore_index"] = 255
    system_dict["lr_scheduler"] = {}
    system_dict["lr_scheduler"]["type"] = lr_scheduler_type


def initialize_train_hyperparameters(
    epochs=80,
    save_period=10,
    early_stop=10,
    monitor="max Mean_IoU",
    tensorboard=True,
    log_dir="saved/runs",
    log_per_iter=20,
    val_per_epochs=5,
    save_dir="saved/",
):

    system_dict["trainer"] = {}
    system_dict["trainer"]["epochs"] = epochs
    system_dict["trainer"]["save_dir"] = save_dir
    system_dict["trainer"]["save_period"] = save_period
    system_dict["trainer"]["monitor"] = monitor
    system_dict["trainer"]["early_stop"] = early_stop
    system_dict["trainer"]["tensorboard"] = tensorboard
    system_dict["trainer"]["log_dir"] = log_dir
    system_dict["trainer"]["log_per_iter"] = log_per_iter
    system_dict["trainer"]["val"] = True
    system_dict["trainer"]["val_per_epochs"] = 5

    with open("config.json", "w") as fp:
        json.dump(system_dict, fp)


if __name__ == "__main__":
    initialize_train_dataloader()
    initialize_val_dataloader()
    initialize_model()
    initialize_optimizer_loss()
    initialize_train_hyperparameters()
