
import os
import sys
sys.path.append('.')
from easydict import EasyDict as edict
import numpy as np
from typing import List, Dict, Optional, Union
import torch
from dataclasses import dataclass
import cv2

from .ocr import OCR
from common_ml.model import FrameModel
from common_ml.tags import FrameTag
from common_ml.types import Data

@dataclass
class RuntimeConfig(Data):
    fps: int
    allow_single_frame: bool
    w_thres: float
    l_thres: bool

    @staticmethod
    def from_dict(data: dict) -> 'RuntimeConfig':
        return RuntimeConfig(**data)

class OCRModel(FrameModel):
    def __init__(self, model_input_path: str, runtime_config: Union[dict, RuntimeConfig]):

        self.model_input_path = model_input_path
        if isinstance(runtime_config, dict):
            self.config = RuntimeConfig.from_dict(runtime_config)
        elif isinstance(runtime_config, RuntimeConfig):
            self.config = runtime_config
        else:
            raise TypeError("runtime_config must be a dict or RuntimeConfig object")
        
        args = self.__add_params()
        device = torch.device('cuda')

        if args.word_rgb:
            args.word_in_channels = 3

        args.weight_folder = os.path.join(self.model_input_path, 'ocr')
        self.ocr_inference = OCR(
            device,
            os.path.join(args.weight_folder, args.craft_model),
            os.path.join(args.weight_folder, args.refiner_model),
            os.path.join(args.weight_folder, args.recognition_model),
            args.text_threshold,
            args.region_threshold,
            args.affinity_threshold,
            args.link_threshold,
            args.magnification_ratio,
            args.magnification_limit, True, True,
            args.insensitive,
            args.keep_ratio_and_pad,
            args.word_rgb,
            args.word_image_width,
            args.word_image_height,
            args.Transformation,
            args.n_fiducial,
            args.FeatureExtraction,
            args.word_in_channels,
            args.feature_extraction_channels,
            args.SequenceModeling,
            args.hidden_size,
            args.Prediction, 
            args.max_text_len
        )
        self.args = args

    def __add_params(self):
        params = edict({
            'weight_folder': '',#'Folder for pretrained weights'
            'image_batch_size': 8,#'Inference batch size for full images'
            'word_batch_size': 256,#'Inference batch size for word images'
            # CRAFT
            'craft_model': 'craft_mlt_25k.pth',#'Pretrained weights for CRAFT'
            'refiner_model': 'craft_refiner_CTW1500.pth',#'Pretrained weights for LinkRefiner'
            'region_threshold': 0.4,#'Determine character region'
            'affinity_threshold': 0.4,#'Determine link between characters'
            'link_threshold': 0.4,#'Determine text-line center line'
            'text_threshold': 0.7,#'Remove non-text regions'
            'magnification_ratio': 1.5,#'Magnification ratio for full images.')
            'magnification_limit': 960,#'Upper limit for both width and height of magnified full images.')
            # SceneTextRecognition
            'recognition_model': 'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth',#'Pretrained weights for recognition model.')
            'word_image_width': 100,#'Resolution for cropped word image'
            'word_image_height': 32,#'Resolution for cropped word image'
            'keep_ratio_and_pad': 0,#'Whether to keep word image aspect ratio then pad for resizing'
            'insensitive': 0,#'36 characters instead of 94'
            'word_rgb': 0,#'Use RGB for cropped word image, instead of grayscale'
            # Stage 1
            'Transformation': 'TPS',#'None | TPS')
            'n_fiducial': 20,#'Number of fiducial points of TPS-STN'
            # Stage 2
            'FeatureExtraction': 'ResNet',#'VGG | RCNN | ResNet'
            'word_in_channels': 1,#'Number of input channels for feature extractor'
            'feature_extraction_channels': 512,#'Number of output channels for feature extractor'
            # Stage 3
            'SequenceModeling': 'BiLSTM',#'None | BiLSTM'
            'hidden_size': 256,#'Size of LSTM hidden state'
            # Stage 4
            'Prediction': 'Attn',#'CTC | Attn'
            'max_text_len': 25,#'Max length (number of characters) of a text instance'
        })
        return params
    
    def _format_as_frame_tag(self, data) -> Dict[int, List[FrameTag]]:
        result = {}
        for frame_idx, item in data.items():
            item = item["ocr"].copy()
            for i in range(len(item["tags"])):
                del item["tags"][i]["polygon"]
            del item["words"]
            result[frame_idx] = [FrameTag(**tag) for tag in item["tags"]]
        return result

    def tag_frames(self, frames: List[np.ndarray]) -> List[FrameTag]:
        ocr_tags = {}
        ocr_batch = 10
        for i in range(0, len(frames), ocr_batch):
            res = self.ocr_inference.inference(
                frames[i:i+ocr_batch], self.args.word_batch_size, self.config.w_thres, self.config.l_thres)
            if res:
                ocr_tags.update(
                    {idx: tag for idx, tag in res.items() if tag})
        ocr_tags = self._format_as_frame_tag(ocr_tags)
        return ocr_tags
    
    def tag(self, frame: np.ndarray) -> List[FrameTag]:
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        tags = self.ocr_inference.inference([frame], self.args.word_batch_size, self.config.w_thres, self.config.l_thres)
        res = []
        for tag in tags[0]['ocr']['tags']:
            box = tag["box"]
            # limit to only x1, x2, y1, y2
            box = {"x1": box["x1"], "x2": box["x2"], "y1": box["y1"], "y2": box["y2"]}
            res.append(FrameTag.from_dict({"text": tag["text"], "confidence": tag["confidence"], "box": box}))
        return res