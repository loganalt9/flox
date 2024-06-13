from base import BaseLogger
from typing import Any
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self) -> None:
        self.records = []
        self.writer = SummaryWriter()
    
    def log(self, name: str, value: Any) -> None:
        ...
    
    def log_dict(self, record: dict) -> None:
        ...
    
    def clear(self) -> None:
        ...