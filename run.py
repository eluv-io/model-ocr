import argparse
import os
import json
from typing import List, Callable
from common_ml.utils import nested_update
from common_ml.model import default_tag, run_live_mode
import setproctitle

from ocr.ocr_model import OCRModel
from config import config

def make_tag_fn(cfg: dict) -> Callable:
    model = OCRModel(config["model_path"], runtime_config=cfg)
    tags_out = os.getenv('TAGS_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags'))
    if not os.path.exists(tags_out):
        os.makedirs(tags_out)

    def tag_fn(file_paths: List[str]) -> None:
        default_tag(model, file_paths, tags_out)

    return tag_fn

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
    setproctitle.setproctitle("model-ocr")
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='*', type=str, default=[])
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--live', action='store_true', default=False)
    args = parser.parse_args()
    
    cfg_raw = json.loads(args.config) if args.config else {}
    default_cfg = config["runtime"]["default"]
    runtime_cfg = nested_update(default_cfg, cfg_raw)
    
    tag_fn = make_tag_fn(runtime_cfg)

    if args.live:
        print('Running in live mode...')
        run_live_mode(tag_fn)
    else:
        tag_fn(args.file_paths)