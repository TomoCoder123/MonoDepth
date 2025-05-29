import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    DPTForSemanticSegmentation,
    pipeline
)