"""CSV logger for experiment metrics."""
import os
import csv
from typing import Dict, Any


class CSVLogger:
    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.fieldnames = None
        self.file = None
        self.writer = None

    def log(self, row: Dict[str, Any]):
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            self.file = open(self.path, "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        else:
            # Extend fieldnames if new keys appear
            new_keys = [k for k in row.keys() if k not in self.fieldnames]
            if new_keys:
                self.fieldnames.extend(new_keys)
                self.file.close()
                # Rewrite with new fieldnames
                with open(self.path, "r", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.file = open(self.path, "w", newline="", encoding="utf-8")
                self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
                self.writer.writeheader()
                self.writer.writerows(rows)
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
