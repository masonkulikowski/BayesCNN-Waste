import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(Path(__file__).parent.parent / config_path) as f:
        return yaml.safe_load(f)