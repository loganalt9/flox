from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class BaseLogger(Protocol):
    records: list

    def __init__(self) -> None:
        self.records = []

    def log(self, name: str, loss: int, round: int | None) -> None:
        self.records.append({'name': name,
                            'train/loss': loss,
                             "round": round})

    def log_dict(self, record: dict) -> None:
        self.records.append(record)