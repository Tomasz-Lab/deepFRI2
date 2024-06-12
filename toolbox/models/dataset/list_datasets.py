import os
import re
from pathlib import Path

import dotenv

dotenv.load_dotenv()
SEPARATOR = os.getenv("SEPARATOR")
data_path = os.getenv('DATA_PATH')

if __name__ == "__main__":

    dataset_path =  f"{data_path}/dataset"

    pattern = f".*\\{SEPARATOR}.*\\{SEPARATOR}.*\\{SEPARATOR}.*"

    pattern = re.compile(pattern)

    dataset_paths = Path(dataset_path).glob("*")

    for path in dataset_paths:
        if path.is_dir() and pattern.match(path.name) is not None:
            print(path.name)
            dataset_name = path.name
            dataset_name = dataset_name.split("-")
            print(f"DB:{dataset_name[0]} type:{dataset_name[1]} type_str:{dataset_name[2]} version:{dataset_name[3]}")
            