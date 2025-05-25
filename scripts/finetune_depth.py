import multiprocessing as mp
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from logger import Logger
# from transformers import get_cosine_schedule_with_warmup

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_depth_model(
        model,
        train_loader,
        val_loader,
        num_epochs = 10,
        learning_rate = 1e-6,
        weight_decay = 0.1,
        warmup_epochs = 1.0,
        save_dir = "models/pretrained",
        size = None,
        device = DEVICE,
        patience = 5,

):
    wandb.init(
        project = "lunar-mono-depth",
        config = {
            "model_size": size,
            "learning_rate": learning_rate,
            "weight_decay", weight_decay,
            "warmup_epochs": warmup_epochs,
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "patience": patience,
        }
    )
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%MS")
    filename = (
        f"mono_depth_{size}_{timestamp}.pth"
        if size is not None
        else f"mono_depth_{size}.pth"
    )
    #freeze encoder parameters
    n_frozen, n_trainable = 0,0
    for name, param in model.model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
            n_frozen += 1
        else: 
            n_trainable += 1
    Logger.info(f"Freezing {n_frozen} parameters, {n_trainable} trainable", name = model.name)
    optimizer = AdamW(
       
    )





