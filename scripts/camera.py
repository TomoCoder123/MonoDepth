from pathlib import Path

import numpy as np
import torch

from scripts.utils import (
    DEVICE,
    FRONT_LEFT_UP_T_OPENCV,
    Config,
    field_of_view2focal_length,
    focal_length2field_of_view,
    load_config,
)
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from thirdparty.monogs.utils.camera_utils import Camera as MonoGS_Camera


class Camera(MonoGS_Camera):
    """Camera class for lunar environment simulation.

    This class extends the MonoGS_Camera class to provide camera functionality
    specific to the lunar environment simulation. It handles camera parameters,
    pose updates, and coordinate transformations between different reference frames.

    The camera supports:
    - Intrinsic parameter management
    - Extrinsic parameter updates
    - Coordinate frame transformations
    - Ground truth pose tracking
    - Projection matrix computation

    The camera can be configured through a YAML file that specifies:
    - Resolution (width, height)
    - Field of view angles
    - Camera position relative to the robot body
    - Enabled/disabled state

    Attributes:
        CAMERA_DICT (dict): Dictionary mapping camera class names to their implementations.
            Used for dynamic camera creation from configuration files.
        config (Config): Camera configuration containing all parameters.
        name (str): Unique identifier for the camera.
        enabled (bool): Whether the camera is active and should be used for rendering.
        fovx (float): Field of view in x direction (radians).
        fovy (float): Field of view in y direction (radians).
        body_T_cam_gt (torch.Tensor): Ground truth transformation from body to camera frame.
        cam_T_world_gt (torch.Tensor): Ground truth transformation from camera to world frame.
        cam_T_world (torch.Tensor): Current transformation from camera to world frame.

    Example:
        >>> config = load_config("configs/camera.yaml")
        >>> camera = Camera(config)
        >>> camera.update_pose(world_T_body)
        >>> intrinsics = camera.get_intrinsics()
    """

    CAMERA_DICT = {}

    @staticmethod
    def from_config(config: Config) -> "Camera":
        """Create a camera instance from a configuration.

        This factory method creates the appropriate camera subclass based on the
        configuration provided. The camera class must be registered in CAMERA_DICT
        before it can be instantiated.

        Args:
            config (Config): Configuration object or path containing camera settings.
                Must include resolution and field of view parameters.

        Returns:
            Camera: An instance of the appropriate camera subclass.

        Raises:
            ValueError: If the specified camera class is not registered.
            ValueError: If the configuration is invalid or missing required fields.

        Example:
            >>> config = {"class": "Camera", "res": [640, 480], "fovx": 60}
            >>> camera = Camera.from_config(config)
        """
        if config is None:
            return None
        config = load_config(config)
        class_name = config.get("class", "Camera")
        if class_name not in Camera.CAMERA_DICT:
            raise ValueError(
                f"Unknown camera class: {class_name}. Available classes: {list(Camera.CAMERA_DICT.keys())}"
            )
        return Camera.CAMERA_DICT[class_name](config)

    def __init__(self, config: Config):
        """Initialize the camera.

        Args:
            config (Config): Configuration object containing camera parameters.
        """
        self.config = config = load_config(config)
        self.name = config.name
        self.enabled = config.get("enabled", True)
        self.device = DEVICE

        W, H = config.W, config.H
        cx, cy = W / 2, H / 2
        if config.fovx is not None:
            self.fovx = config.fovx
            fx = field_of_view2focal_length(W, self.fovx)
            fy = fx
            self.fovy = focal_length2field_of_view(H, fy)
        else:
            fx, fy = config.fx, config.fy
            self.fovx = focal_length2field_of_view(W, fx)
            self.fovy = focal_length2field_of_view(H, fy)

        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)

        self.body_T_cam_gt = (
            torch.tensor(config.body_T_cam).float().to(self.device)
            if config.body_T_cam is not None
            else torch.eye(4, device=self.device)
        )
        self.cam_T_world_gt = torch.eye(4, device=self.device)
        self.cam_T_world = torch.eye(4, device=self.device)

        super().__init__(
            uid=self.name,
            color=None,
            depth=None,
            gt_T=self.cam_T_world_gt,
            projection_matrix=projection_matrix,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            fovx=self.fovx,
            fovy=self.fovy,
            image_height=H,
            image_width=W,
            device=self.device,
        )
        self.update_pose(torch.eye(4, device=self.device))
        self.world_T_cam = torch.linalg.inv(self.cam_T_world)
        self.world_T_cam_gt = torch.linalg.inv(self.cam_T_world_gt)

    @classmethod
    def register(cls, name: str):
        """Register a camera class.

        Args:
            name (str): The name to register the camera class under.
        """
        Camera.CAMERA_DICT[name] = cls

    def update_pose_gt(self, world_T_body_gt):
        """Update the ground truth camera pose.

        This method updates the camera's ground truth pose based on the provided
        world-to-body transformation. It computes the resulting camera-to-world
        transformation using the camera's fixed offset from the body frame.

        Args:
            world_T_body_gt (torch.Tensor or np.ndarray): Ground truth transformation
                from world to body frame. Shape (4, 4).

        Example:
            >>> world_T_body_gt = torch.eye(4)  # Identity transformation
            >>> camera.update_pose_gt(world_T_body_gt)
        """
        if not isinstance(world_T_body_gt, torch.Tensor):
            world_T_body_gt = torch.from_numpy(world_T_body_gt)
        world_T_body_gt = world_T_body_gt.float().to(self.device)
        world_T_cam_gt = world_T_body_gt @ self.body_T_cam_gt
        world_T_cam_ocv_gt = world_T_cam_gt @ FRONT_LEFT_UP_T_OPENCV.to(self.device)
        camera_ocv_T_world_gt = torch.linalg.inv(world_T_cam_ocv_gt)

        self.cam_T_world_gt = torch.linalg.inv(world_T_cam_gt)
        self.world_T_cam_gt = torch.linalg.inv(self.cam_T_world_gt)
        self.R_gt = camera_ocv_T_world_gt[:3, :3]
        self.T_gt = camera_ocv_T_world_gt[:3, 3]

    def update_pose(self, world_T_body):
        """Update the current camera pose.

        This method updates the camera's current pose based on the provided
        world-to-body transformation. It computes the resulting camera-to-world
        transformation using the camera's fixed offset from the body frame.

        Args:
            world_T_body (torch.Tensor or np.ndarray): Current transformation
                from world to body frame. Shape (4, 4).

        Example:
            >>> world_T_body = torch.eye(4)  # Identity transformation
            >>> camera.update_pose(world_T_body)
        """
        if not isinstance(world_T_body, torch.Tensor):
            world_T_body = torch.from_numpy(world_T_body)
        world_T_body = world_T_body.float().to(self.device)

        self.world_T_cam = world_T_body @ self.body_T_cam_gt
        self.cam_T_world = torch.linalg.inv(self.world_T_cam)

        ocv_T_flu = torch.linalg.inv(FRONT_LEFT_UP_T_OPENCV.to(self.device))
        cam_ocv_T_world = ocv_T_flu @ self.cam_T_world

        self.R = cam_ocv_T_world[:3, :3]
        self.T = cam_ocv_T_world[:3, 3]

        self.clean()

    def get_intrinsics(self):
        """Get the camera intrinsic parameters.

        Returns a dictionary containing all camera intrinsic parameters needed
        for projection and rendering operations.

        Returns:
            dict: Dictionary containing camera intrinsic parameters:
                - fx (float): Focal length in x direction (pixels)
                - fy (float): Focal length in y direction (pixels)
                - cx (float): Principal point x coordinate (pixels)
                - cy (float): Principal point y coordinate (pixels)
                - W (int): Image width (pixels)
                - H (int): Image height (pixels)
                - fovx (float): Field of view in x direction (radians)
                - fovy (float): Field of view in y direction (radians)

        Example:
            >>> intrinsics = camera.get_intrinsics()
            >>> print(f"Focal length: {intrinsics['fx']}")
        """
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "W": self.image_width,
            "H": self.image_height,
            "fovx": self.fovx,
            "fovy": self.fovy,
        }


# Register the Camera class with Camera
Camera.register("Camera")


def create_camera(time, cam_ocv_T_world, img, intrinsics: dict):
    """Create a new camera viewpoint.

    Args:
        time (float): Timestamp of the viewpoint.
        cam_T_world (np.ndarray): Camera to world transformation matrix.
        img (torch.Tensor): RGB image from the camera.
        intrinsics (dict): Camera intrinsic parameters.

    Returns:
        MonoGS_Camera: A camera viewpoint object.
    """
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
        W=intrinsics["W"],
        H=intrinsics["H"],
    ).transpose(0, 1)

    viewpoint = MonoGS_Camera(
        time,
        img,
        None,
        cam_ocv_T_world,
        projection_matrix,
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
        intrinsics["fovx"],
        intrinsics["fovy"],
        intrinsics["H"],
        intrinsics["W"],
        device=DEVICE,
    )
    viewpoint.update_RT(viewpoint.R_gt.detach().clone(), viewpoint.T_gt.detach().clone())

    return viewpoint