from base import BaseLogger
from typing import Any
import pandas as pd
from pathlib import Path

class CSVLogger:
    def __init__(self) -> None:
        self.records = []

    def log(self, name: str, value: Any) -> None:
        self.records.append({name: value})
    
    def log_dict(self, record: dict) -> None:
        self.records.append(record)

    def _create_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.records)

    def export(self, filename: str | Path) -> None:
        df = self._create_dataframe()
        # Create nicer format to ouput to file
        def format(row):
            return " - ".join([f"{key}: {value}" for key, value in row.items()])
        
        df = df.apply(format, axis=1)
        df.to_csv(filename, index=False, header=False)



log = CSVLogger()
log.log_dict({"id": 1, "loss": 4.3, "epoch": 3, "time": "6pm", "parent": 5})
log.log_dict({"id": 2, "loss": 2.4, "epoch": 2, "time": "2pm", "parent": 4})
log.log_dict({"id": 3})

log2: BaseLogger = log

path = Path('./test.csv')

log.export(path)

log2.clear()

log2.export("./test2.csv")

