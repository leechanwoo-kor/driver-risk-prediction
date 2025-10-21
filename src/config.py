from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Cfg:
    random_seed: int
    paths: dict
    columns: dict
    model: dict

    @property
    def data_dir(self) -> Path: return Path(self.paths["data_dir"])
    @property
    def model_dir(self) -> Path: return Path(self.paths["model_dir"])

def load_config(path: str | Path) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Cfg(**raw)  # type: ignore
