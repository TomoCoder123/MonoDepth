import numpy as np
import torch
import torch.nn.functional as F
from transformers import ( #hugging face transformers libraray
    AutoImageProcessor,   #selects image preprocessor
    AutoModelForDepthEstimation, # loads model architecture pre-trained for depth estimation
    DPTForSemanticSegmentation,  #semantic segmentation model
    pipeline  #preprocessing, inference, and post processing into a single function
)
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
        
