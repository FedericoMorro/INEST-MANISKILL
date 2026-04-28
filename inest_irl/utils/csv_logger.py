"""
Utility for logging metrics to a CSV file during training and evaluation.
It supports both logging and retrieving data.
"""


from absl import logging
import os
from typing import Any


class CSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.fieldnames = []
        self.data = {}
        
    #* LOGGING API
    def init_logging(self, fieldnames: list[str]):
        self.fieldnames = fieldnames
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(self.fieldnames) + "\n")
        self.reset()
            
    def reset(self):
        self.data = {field: None for field in self.fieldnames}    
        
    def flush(self):
        with open(self.csv_path, "a", encoding="utf-8") as f:
            row = [str(self.data[field]) for field in self.fieldnames]
            f.write(",".join(row) + "\n")
        self.reset()
            
    def log(self, data: dict):
        for field in data:
            if field not in self.fieldnames:
                logging.warning(f"Field {field} not in CSVLogger fieldnames; skipping.")
                continue
            if self.data[field] is not None:
                logging.warning(f"Field {field} already has a value; overwriting.")
            self.data[field] = data[field]
            
    #* RETRIEVAL API
    def retrieve_data(self) -> dict[str, list[str]]:
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file {self.csv_path} not found.")
            
        with open(self.csv_path, "r", encoding="utf-8") as f:
            # retrieve header and initialize fieldnames
            header = f.readline().strip().split(",")
            self.fieldnames = header
            self.retrieval_data = {field: [] for field in self.fieldnames}
            # retrieve data
            for line in f:
                values = line.strip().split(",")
                for field, value in zip(header, values):
                    self.retrieval_data[field].append(value)
                        
        return self.retrieval_data
    
    def get_field_data(self, field: str, data_type: type = str) -> list[Any]:
        if not hasattr(self, "retrieval_data"):
            self.retrieve_data()
        if field not in self.fieldnames:
            logging.warning(f"Field {field} not in CSVLogger fieldnames; returning empty list.")
            return []
        return [data_type(value) for value in self.retrieval_data[field]]
    
    def get_latest_value(self, field: str, data_type: type = str) -> Any:
        field_data = self.get_field_data(field, data_type)
        if not field_data:
            logging.warning(f"No data found for field {field}; returning None.")
            return None
        return field_data[-1]
    