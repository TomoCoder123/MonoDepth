import sys
import time

from tqdm import tqdm
class Colors:
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    WHITE = "\033[37m"
class Logger:
    time_start = time.time()
    time_color = Colors.CYAN
    name_color = Colors.GREEN
    
    @staticmethod
    def success(message: str, name: str = None, time_sim: float = None):
        Logger.log(message, name, time_sim, style=Colors.GREEN)

    @staticmethod
    def info(message: str, name: str = None, time_sim: float = None):
        Logger.log(message, name, time_sim, style=Colors.WHITE)

    @staticmethod
    def warning(message: str, name: str = None, time_sim: float = None):
        Logger.log(message, name, time_sim, style=Colors.YELLOW)

    @staticmethod
    def error(message: str, name: str = None, time_sim: float = None):
        Logger.log(message, name, time_sim, style=Colors.RED)

    @staticmethod
    def debug(message: str, name: str = None, time_sim: float = None):
        Logger.log(message, name, time_sim, style=Colors.BLUE)
    
    @staticmethod
    def reset_start_time():
        Logger.time_start = time.time()