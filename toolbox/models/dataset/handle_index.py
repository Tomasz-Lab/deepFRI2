import json
from pathlib import Path
from typing import Dict, Union, List


def read_index(index_file_path: Path) -> Dict[str, str]:
    with index_file_path.open('r') as f:
        index = json.load(f)
        return index


def create_index(index_file_path: Path, values: Union[Dict[str, str], List[str]]):
    with index_file_path.open('w') as f:
        if isinstance(values, list):
            values = {str(i): v for i, v in enumerate(values)}
        json.dump(values, f)
