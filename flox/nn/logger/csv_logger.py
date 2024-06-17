from flox.nn.logger.base import BaseLogger
from datetime import datetime
from typing import Any
import pandas as pd
from pathlib import Path
from flox.flock.node import FlockNode


class CSVLogger:
    def __init__(self, node: FlockNode | None = None) -> None:
        self.records = []

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
    
    def log_dict(self, record: dict) -> None:
        self.records.append(record)

    def export(self, filename: str | Path | None = None) -> None | str:
        def _create_dataframe(records: list) -> pd.DataFrame:
                return pd.DataFrame.from_records(records)
        
        df = _create_dataframe(self.records)
        return df.to_csv(filename, index=False)
