from datetime import datetime
from flox.nn.logger.base import BaseLogger
from typing import Any
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self) -> None:
        self.records = []
        #add param path to logs
        self.writer = SummaryWriter(log_dir='./runs')
    
    def log(
            self, 
            name: str, 
            loss: int,
            round: int | None,
        ) -> None:
        self.records.append({'name': name,
                            'train/loss': loss,
                            'round': round})
        self.writer.add_scalar(name, value, global_step=round)

    def log_dict(self, record: dict) -> None:
        self.records.append(record)
        self.writer.add_scalar(record['name'], record['value'], record['round'])
    
    def clear(self) -> None:
        self.records = []
        self.writer.close()



