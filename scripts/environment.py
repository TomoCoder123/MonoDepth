from typing import Dict, Type
from scripts.utils import Config, load_config

class Environment:
    """Base class for lunar environment simulation.

    This class provides the foundation for creating and managing lunar environments,
    including terrain, lighting, and rendering capabilities. It serves as an abstract
    base class that defines the interface for environment implementations.

    The environment supports:
    - Terrain generation and modification
    - Dynamic lighting conditions
    - Camera-based rendering
    - Agent pose tracking
    - Collision detection
    - Height map queries

    Attributes:
        ENVIRONMENT_DICT (Dict[str, Type["Environment"]]): Dictionary mapping environment
            class names to their implementations. Used for dynamic environment creation
            from configuration files.

    Example:
        >>> config = load_config("configs/env.yaml")
        >>> env = Environment.from_config(config)
        >>> camera = Camera(config.camera)
        >>> rendered = env.render(camera)
    """

    ENVIRONMENT_DICT: Dict[str, Type["Environment"]] = {}

    @staticmethod
    def from_config(config: Config) -> "Environment":
        """Create an environment instance from a configuration.

        This factory method creates the appropriate environment subclass based on the
        configuration provided. The environment class must be registered in ENVIRONMENT_DICT
        before it can be instantiated.

        Args:
            config (Config): Configuration object or path containing environment settings.
                Must include a 'class' field specifying the environment type.

        Returns:
            Environment: An instance of the appropriate environment subclass.

        Raises:
            ValueError: If the specified environment class is not registered.
            ValueError: If the configuration is invalid or missing required fields.

        Example:
            >>> config = {"class": "Open3DEnv", "surface": {...}}
            >>> env = Environment.from_config(config)
        """
        if config is None:
            return None
        config = load_config(config)
        class_name = config["class"]
        if class_name not in Environment.ENVIRONMENT_DICT:
            raise ValueError(
                f"Unknown environment class: {class_name}. Available classes: {list(Environment.ENVIRONMENT_DICT.keys())}"
            )
        return Environment.ENVIRONMENT_DICT[class_name](config)