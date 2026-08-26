import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from vision.segmentation.config import Config
from vision.segmentation.datasets.oxford_pet import load_dataloaders
from vision.segmentation.models.unet import UNet
from vision.segmentation.training.evaluation import test_evaluation
from vision.segmentation.training.train import train
from vision.segmentation.utils.visualization import get_device

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "vision" / "segmentation" / "config.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Hello from torch-foundry!")
    config = Config.from_json(CONFIG_PATH)
    set_seed(config.seed)

    data_root = Path(config.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    # 1: loade train/val/test loader
    train_loader, val_loader, test_loader = load_dataloaders(
        root=data_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
        val_ratio=config.val_ratio,
    )
    print("train", len(train_loader), "val", len(val_loader), "test", len(test_loader))

    # 2: get one_batch for check out compute model output shape
    # images, masks = next(iter(train_loader))
    device = get_device()
    print("device:", device)
    model = UNet(3, config.num_classes)
    model.to(device)
    total_params = sum([p.numel() for p in model.parameters()])
    print("total params:", total_params)

    # 3: criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    output_dir = (
        PROJECT_ROOT
        / "vision"
        / "segmentation"
        / "runs"
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print("output directory:", output_dir)
    effective_config = config.to_dict()
    effective_config["data_root"] = str(data_root)
    (output_dir / "config.json").write_text(
        json.dumps(effective_config, indent=2) + "\n", encoding="utf-8"
    )

    # training start
    _history = train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config.num_epochs,
        num_classes=config.num_classes,
        output_dir=output_dir,
    )

    # test start
    _, _, _, _, _ = test_evaluation(
        test_loader, model, criterion, device, num_classes=config.num_classes
    )


if __name__ == "__main__":
    main()
