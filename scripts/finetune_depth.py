import multiprocessing as mp
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from tqdm import tqdm
from scripts.logger import Logger
from transformers import get_cosine_schedule_with_warmup

import wandb
from models.depth import DepthModel
from models.depth import ScaleInvariantLoss, compute_depth_metrics
from scripts.environment import Environment
from scripts.camera import Camera
from scripts.utils import read_pfm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_config = "configs/env.yaml::open3d_small"
cam_config = dict(
    H=360,  # [px] Image height
    W=640,  # [px] Image width
    fovx=1.2217304763960306,  # [rad]
)

# Dataset parameters
height_lims = [0.5, 2.0]  # [m] Camera height
azimuth_lims = [0, 360.0]  # [deg] Sun azimuth
elevation_lims = [10, 90.0]  # [deg] Sun elevation
xlims = [-30, 30]  # [m] X limits
ylims = [-20, 20]  # [m] Y limits
n_train = 5000  # Increased from 1000
n_val = 100  # Increased from 20
def visualize_depth(depth, vmin=None, vmax=None):
    """Convert depth map to RGB visualization."""
    # Create a copy to avoid modifying the original
    depth_viz = depth.copy()

    # Replace NaN values with max value for visualization
    if np.isnan(depth_viz).any():
        valid_max = np.nanmax(depth_viz)
        depth_viz[np.isnan(depth_viz)] = valid_max

    if vmin is None:
        vmin = np.nanmin(depth_viz)
    if vmax is None:
        vmax = np.nanmax(depth_viz)

    # Normalize to [0, 1]
    depth_norm = np.clip(depth_norm, 0, 1)

    # Apply colormap
    cmap = plt.cm.viridis
    depth_colored = cmap(depth_norm)[..., :3]  # Remove alpha channel

    # Create a mask for NaN values
    nan_mask = np.isnan(depth)
    if nan_mask.any():
        # Set NaN regions to black in the visualization
        depth_colored[nan_mask] = [0, 0, 0]

    return (depth_colored * 255).astype(np.uint8)
def create_visualization_grid(rgb, depth_gt, depth_pred, max_samples=4):
    """Create a vertical concatenation of RGB, GT depth, and predicted depth images."""
    # Convert tensors to numpy
    rgb_np = rgb[:max_samples].cpu().permute(0, 2, 3, 1).numpy()  # BCHW -> BHWC
    depth_gt_np = depth_gt[:max_samples].cpu().numpy()
    depth_pred_np = depth_pred[:max_samples].detach().cpu().numpy()

    # Ensure RGB is in [0, 1] range
    rgb_np = np.clip(rgb_np, 0, 1)

    # Convert to uint8
    rgb_np = (rgb_np * 255).astype(np.uint8)

    # Visualize depths
    depth_gt_viz = np.stack([visualize_depth(d) for d in depth_gt_np])
    depth_pred_viz = np.stack([visualize_depth(d) for d in depth_pred_np])

    # Concatenate vertically for each sample
    samples = []
    for i in range(len(rgb_np)):
        sample = np.concatenate([rgb_np[i], depth_gt_viz[i], depth_pred_viz[i]], axis=0)
        samples.append(sample)

    # Concatenate samples horizontally
    grid = np.concatenate(samples, axis=1)

    return grid

class LuDataset(Dataset):
    def __init__(self, filenames):
        self.img_paths = [os.path.join(img_dir, f) for f in filenames]
        self.mask_paths = [os.path.join(mask_dir, f) for f in filenames]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # img = cv2.imread(self.img_paths[idx])
        # mask = cv2.imread(self.mask_paths[idx])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # img_tensor = to_tensor(resize(Image.fromarray(img), (256, 256)))
        # mask_class = rgb_to_class(mask)
        # mask_tensor = torch.from_numpy(
        #     np.array(resize(Image.fromarray(mask_class), (256, 256), Image.NEAREST))
        # ).long()
        return img_tensor, mask_tensor
class Open3DDataset(Dataset):
    def __init__(
        self, n_images, cam_config, env_config, xlims, ylims, hlims, azlims, ellims, mode="train"
    ):
        self.n_images = n_images
        self.mode = mode
        self.is_train = mode == "train"

        # Generate random poses and sun positions
        xs = np.random.uniform(xlims[0], xlims[1], n_images)
        ys = np.random.uniform(ylims[0], ylims[1], n_images)
        thetas = np.random.uniform(0, 2 * np.pi, n_images)
        hs = np.random.uniform(hlims[0], hlims[1], n_images)
        self.elevs = np.random.uniform(ellims[0], ellims[1], n_images)
        self.azims = np.random.uniform(azlims[0], azlims[1], n_images)

        # Store configuration for lazy initialization
        self.cam_config = cam_config
        self.env_config = env_config
        self.poses = [np.array([x, y, theta, h]) for x, y, theta, h in zip(xs, ys, thetas, hs)]

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # Lazy initialization of environment and camera
        if not hasattr(self, "env"):
            self.env = Environment.from_config(self.env_config)
        if not hasattr(self, "cam"):
            self.cam = Camera(self.cam_config)

        # Get pose and update camera
        x, y, theta, h = self.poses[idx]
        pose = self.env.get_agent_pose(np.array([x, y, theta]), h=h)
        self.cam.update_pose(pose)
        self.env.set_sun_position(self.azims[idx], self.elevs[idx])
        rendered = self.env.render(self.cam)

        # Convert to numpy first, then to tensor to ensure CPU tensors
        # Normalize RGB to [0, 1] range
        img = torch.from_numpy(rendered["rgb"].astype(np.float32) / 255.0).permute(
            2, 0, 1
        )  # HWC -> CHW

        # Get depth values directly from renderer
        depth = torch.from_numpy(rendered["depth"].astype(np.float32))

        # Downscale to largest multiple of 32
        h = 32 * (img.shape[1] // 32)
        w = 32 * (img.shape[2] // 32)
        img = F.interpolate(img.unsqueeze(0), (h, w), mode="bilinear", align_corners=False).squeeze(
            0
        )
        depth = (
            F.interpolate(
                depth.unsqueeze(0).unsqueeze(0), (h, w), mode="bilinear", align_corners=False
            )
            .squeeze(0)
            .squeeze(0)
        )

        # Normalize depth to [0, 1]
        mask = torch.isfinite(depth)
        if mask.any():
            depth_max = torch.max(depth[mask].reshape(-1))
            depth_min = torch.min(depth[mask].reshape(-1))
            depth = (depth - depth_min) / (depth_max - depth_min)

        return img, depth
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
            "weight_decay": weight_decay,
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
        [
            {"params": [
                param
                for name, param in model.model.named_parameters()
                if "neck" in name and param.requires_grad
            ], 
            "lr": learning_rate *5,
            }, {
                "params": [
                    param
                    for name, param in model.model.named_parameters()
                    if "head" in name and param.requires_grad
                ],
                "lr": learning_rate * 10,
            },
        ], 
        lr = learning_rate,
        weight_decay = weight_decay,
    )

    #Setup scheduler with longer warmup and cosine decay
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_training_steps)
    criterion = ScaleInvariantLoss()
    scaler = torch.amp.GradScaler()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Training Loop
    for epoch in range(num_epochs):
        model.model.train()
        train_loss = 0
        train_metrics=  { 
            "abs_rel": 0.0,
            "rmse": 0.0,
            "rmse_log": 0.0,
            "a1": 0.0,
            "a2": 0.0,
            "a3": 0.0,
            }
        for batch_idx, (rgb, depth_gt) in enumerate(tqdm(train_loader, desc=f"Epoch{epoch+1}")):
            rgb = rgb.to(device)     
            depth_gt = depth_gt.to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                depth_pred = model.compute_depth(rgb)
                loss = criterion(depth_pred, depth_gt)
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm =1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            metrics = compute_depth_metrics(depth_pred, depth_gt)
            for k, v in metrics.items():
                train_metrics[k] += v
            train_loss += loss.item()
            if batch_idx == 0:
                grid = create_visualization_grid(rgb, depth_gt, depth_pred)
                wandb.log(
                    {
                        "train/visualization": wandb.Image(grid),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    }
                )
        train_loss /= len(train_loader)
        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
        model.model.eval()
        val_loss = 0
        val_metrics = {
            "abs_rel": 0.0,
            "rmse": 0.0,
            "rmse_log": 0.0,
            "a1": 0.0,
            "a2": 0.0,
            "a3": 0.0,
        }
        with torch.no_grad():
            for batch_idx, (rgb, depth_gt) in enumerate(tqdm(val_loader, desc = "Validation")):
                rgb = rgb.to(device)
                depth_gt = depth_gt.to(device)
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    depth_pred = model.compute_depth(rgb)
                    loss = criterion(depth_pred, depth_gt)

                metrics = compute_depth_metrics(depth_pred, depth_gt)
                for k, v in metrics.items():
                    val_metrics[k] += v

                val_loss += loss.item()

                if batch_idx == 0:
                    grid = create_visualization_grid(rgb, depth_gt, depth_pred)
                    wandb.log({"val/visualization": wandb.Image(grid)})
        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}

        # Log metrics to wandb
        wandb.log(
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "epoch": epoch + 1,
            }
        )
        # Print results
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print("Train Metrics:", {k: f"{v:.4f}" for k, v in train_metrics.items()})
        print(f"Val Loss: {val_loss:.4f}")
        print("Val Metrics:", {k: f"{v:.4f}" for k, v in val_metrics.items()})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.model.state_dict().copy()
            save_path = os.path.join(save_dir, filename) ### make sure I get this right
            torch.save(best_model_state, save_path)
            print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered")
                # Restore best model
                model.model.load_state_dict(best_model_state)
                break
    val_loss /= len(val_loader)
    val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}

    # Log metrics to wandb
    wandb.log({"val/loss": val_loss, **{f"val/{k}": v for k, v in val_metrics.items()}})

    # Close wandb
    wandb.finish()
    
    #Next Steps: Finish the training function, understand how the model is created and works,
    # Create the dataset, find a way to obtain the lusnar dataset, import the model mono4Depth

def main():
    # Configuration ???where does this come from
    
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../shared/data_raw/LuSNAR/Moon_1/image0/'))

    rgb_dir = Path(os.path.join(data_path,'rgb'))
    depth_dir = Path(os.path.join(data_path, 'depth'))
    # !!!Create but for the other datset.
    print(depth_dir)
    dataset = "LuSNAR"
    if dataset=="LuSNAR":
        
        root_dir = Path(data_path)
        data_list = []
        count = 0

        for file in depth_dir.glob("*.pfm"):    
            added_path = f"{file}"
            absolute_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__),added_path))
            data_list.append(absolute_filepath)
        print(data_list[0])
        image = read_pfm(data_list[0])
        # print(image)
        print("plotting")
        plt.plot([1,2,3,2,4])
        plt.show()
        plt.imshow(image)
        plt.show()


    if dataset=="Open3D":
        train_ds = Open3DDataset( 
            n_images=n_train,
            cam_config=cam_config,
            env_config=env_config,
            xlims=xlims,
            ylims=ylims,
            hlims=height_lims,
            azlims=azimuth_lims,
            ellims=elevation_lims,
            mode="train",
        )
        val_ds = Open3DDataset(
            n_images=n_val,
            cam_config=cam_config,
            env_config=env_config,
            xlims=xlims,
            ylims=ylims,
            hlims=height_lims,
            azlims=azimuth_lims,
            ellims=elevation_lims,
            mode="val",
        )
        config = {
            "class": "DepthTransformer",
            "model": "depth_anything_v2",
            "size": "small",
        }
     # Create dataloaders with increased batch size and workers
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    model = DepthModel.from_config(config)
    model.model.train() 
    train_depth_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=5e-7,  # Reduced from 1e-6 to better preserve pretrained features
        weight_decay=0.1,  # Keep weight decay to prevent overfitting
        warmup_epochs=1.0,  # Keep warmup for stable training
        save_dir="pretrained",
        size=config["size"],
        patience=5,
    )

if __name__ == "__main__":
    main()



