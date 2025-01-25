
import argparse
import os
import json
from typing import List
from common_ml.utils import nested_update
from common_ml.model import default_tag

from ocr.ocr_model import OCRModel
from config import config

# Generate tag files from a list of video/image files and a runtime config
# Runtime config follows the schema found in celeb.model.RuntimeConfig
def run(file_paths: List[str], runtime_config: str=None):
    if runtime_config is None:
        cfg = config["runtime"]["default"]
    else:
        cfg = json.loads(runtime_config)
        cfg = nested_update(config["runtime"]["default"], cfg)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    model = OCRModel(config["model_path"], runtime_config=cfg)
    default_tag(model, file_paths, out_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='+', type=str)
    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    run(args.file_paths, args.config)