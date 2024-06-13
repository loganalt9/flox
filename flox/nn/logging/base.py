from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class BaseLogger(Protocol):
    records: list

    def __init__(self) -> None:
        self.records = []

    def log(self, name: str, value: Any) -> None:
        self.records.append({name: value})

    def log_dict(self, record: dict) -> None:
        self.records.append(record)