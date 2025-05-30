import os 
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ( #hugging face transformers libraray
    AutoImageProcessor,   #selects image preprocessor
    AutoModelForDepthEstimation, # loads model architecture pre-trained for depth estimation
    DPTForSemanticSegmentation,  #semantic segmentation model
    pipeline  #preprocessing, inference, and post processing into a single function
)

from scripts.utils import Config, load_config
class ScaleInvariantLoss(torch.nn.Module):
    """Scale and shift invariant loss for depth estimation.

    This loss function is invariant to scale and shift transformations,
    making it suitable for training depth models on datasets with different scales.
    """     
    def __init__(self, alpha=0.5, beta=0.5):
        """Initialize the loss function.

        Args:
            alpha: Weight for the scale-invariant term
            beta: Weight for the shift-invariant term
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target, mask=None):   

            """Compute the scale and shift invariant loss.

            Args:
                pred: Predicted depth map
                target: Ground truth depth map
                mask: Optional mask for valid pixels

            Returns:
                torch.Tensor: Loss value
            """
            #Ensure positive inputs
            pred = torch.clamp(pred, min=1e-6)
            target = torch.clamp(target, min=1e-6)

            # Convert to log space
            pred_log = torch.log(pred)
            target_log = torch.log(target)

            mask = ~torch.isnan(target) & ~torch.isinf(target)
            n_valid = torch.sum(mask) + 1e-6

            #Compute scale-invariant loss
            diff = pred_log - target_log
            diff = torch.where(mask, diff, torch.zeros_like(diff))

            # Compute mean squared error
            mse = torch.sum(diff * diff)/ n_valid
            mean_diff = torch.sum(diff) / n_valid
            # Final Loss
            loss = mse - self.alpha * (mean_diff * mean_diff)
            return loss
            
    def compute_depth_metrics(pred, target):
        """Compute depth estimation metrics.

        Args:
            pred: Predicted depth map
            target: Ground truth depth map
            mask: Optional mask for valid pixels

        Returns:
            dict: Dictionary containing metrics
        """
        # Ensure inputs are positive
        pred = torch.clamp(pred, min=1e-6)
        target = torch.clamp(target, min=1e-6)
        mask = ~torch.isnan(target) & ~torch.isinf(target)
        n_valid = torch.sum(mask) + 1e-6

        diff = torch.where(mask, pred - target, torch.zeros_like(pred))
        log_diff = torch.where(mask, torch.log(pred) - torch.log(target), torch.zeros_like(pred))
        target_masked = torch.where(mask, target, torch.ones_like(target))

        # Compute metrics
        abs_rel = torch.sum(torch.abs(diff) / target_masked) / n_valid
        rmse = torch.sqrt(torch.sum(diff * diff) / n_valid)
        rmse_log = torch.sqrt(torch.sum(log_diff * log_diff) / n_valid)

        # Threshold accuracy
        thresh = torch.maximum((target / pred), (pred / target))
        a1 = torch.sum((thresh < 1.25).float()) / n_valid
        a2 = torch.sum((thresh < 1.25**2).float()) / n_valid
        a3 = torch.sum((thresh < 1.25**3).float()) / n_valid

        return {
            "abs_rel": abs_rel.item(),
            "rmse": rmse.item(),
            "rmse_log": rmse_log.item(),
            "a1": a1.item(),
            "a2": a2.item(),
            "a3": a3.item(),
        }

     
class DepthModel:
    """Base class for depth estimation models.

    This class provides the foundation for implementing different depth estimation models
    for the lunar environment. It defines the interface for depth computation.

    Attributes:
        DEPTH_MODEL_DICT (dict): Dictionary mapping depth model class names to their implementations.
        config (Config): Model configuration.
    """
    DEPTH_MODEL_DICT = {}
    def __init__(self, config: Config):
        self.config = load_config(config)
        self.model :torch.nn.Model = None
    def compute_depth(self, *args, **kwargs):
        raise NotImplementedError("subclasses must implement")
    @staticmethod
    def from_config(config: Config) -> "DepthModel":
        """Create a depth model instance from a configuration.

        Args:
            config (Config): Configuration object or path containing model settings.

        Returns:
            DepthModel: An instance of the appropriate depth model subclass.

        Raises:
            ValueError: If the specified depth model class is not registered.
        """
        if config is None:
            return None
        config = load_config(config)
        class_name = config["class"]
        if class_name not in DepthModel.DEPTH_MODEL_DICT:
            raise ValueError(
                f"Unknown depth model class: {class_name}. Available classes: {list(DepthModel.DEPTH_MODEL_DICT.keys())}"
            )
        return DepthModel.DEPTH_MODEL_DICT[class_name](config)
    @classmethod
    def register(cls, name: str):
        """Register a depth model class.

        Args:
            name (str): The name to register the depth model class under.
        """
        DepthModel.DEPTH_MODEL_DICT[name] = cls
    def compute_depth(self, *args, **kwargs):
            raise NotImplementedError("Subclasses must implement this method.")
        
        
