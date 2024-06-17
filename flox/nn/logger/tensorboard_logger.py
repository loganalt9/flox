from datetime import datetime
from flox.nn.logger.base import BaseLogger
from typing import Any
from torch.utils.tensorboard import SummaryWriter
from flox.flock.node import FlockNode, NodeKind

class TensorBoardLogger:
    def __init__(self, node: FlockNode | None = None) -> None:
        self.records = []
        
        if node: 
            self.writer = SummaryWriter(log_dir=f'./runs/{node.idx}')
        else:
            self.writer = SummaryWriter(log_dir='./runs')

    def log(
            self, 
            name: str, 
            value: Any,
            nodeid: str | None = None,
            epoch: int | None = None,
            time: datetime | None = None
    ) -> None:

        self.records.append({'name': name,
                            'value': value,
                            'nodeid': nodeid,
                            'epoch': epoch,
                            'datetime': time or datetime.now()})

        self.writer.add_scalar(name, value, global_step=epoch, walltime=time)

    def log_dict(self, record: dict) -> None:
        self.records.append(record)
        self.writer.add_scalar(record['name'], record['value'], global_step=record['epoch'], walltime=record['datetime'])
    
    def clear(self) -> None:
        self.records = []
        self.writer.close()
    
"""
node1 = FlockNode(idx='ag1', kind=NodeKind.AGGREGATOR)
node2 = FlockNode(idx='work1', kind=NodeKind.WORKER)

logger1 = TensorBoardLogger(node1)
logger2 = TensorBoardLogger(node2)

import numpy as np
for step in range(100):
    logger1.log('Train/Loss', np.random.random(), 'node 1', step)
    logger2.log('Train/Loss', np.random.random(), 'node 2', step)
"""