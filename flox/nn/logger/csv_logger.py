from flox.nn.logger.base import BaseLogger
from typing import Any
import pandas as pd
from pathlib import Path

class CSVLogger:
    def __init__(self) -> None:
        self.records = []

    def log(self, name: str, loss: int, round: int | None) -> None:
        self.records.append({'name': name,
                            'train/loss': loss,
                            'round': round})
    
    def log_dict(self, record: dict) -> None:
        self.records.append(record)

    def export(self, filename: str | Path | None) -> None | str:
        def _create_dataframe(records: list) -> pd.DataFrame:
                return pd.DataFrame.from_records(records)
        
        df = _create_dataframe(self.records)
        return df.to_csv(filename, index=False)

