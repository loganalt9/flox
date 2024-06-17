from typing import Protocol, Any, runtime_checkable
from datetime import datetime
from flox.flock.node import FlockNode

@runtime_checkable
class BaseLogger(Protocol):
    records: list

    def __init__(self, node: FlockNode | None = None) -> None:
        self.records = []

    def log(
            self, 
            name: str,
            value: Any,
            nodeid: str | None,
            epoch: int | None,
            time: datetime | None
    ) -> None:
        """
        args:
            - name: type to be logged
            - value: the value of the type to be logged
            - nodeid: identifies which node the log pertains to
            - epoch: training round
            - time: time of log
        """
        self.records.append({'name': name,
                            'value': value,
                            'nodeid': nodeid,
                            'epoch': epoch,
                            'datetime': time or datetime.now()})

    def log_dict(self, record: dict) -> None:
        self.records.append(record)