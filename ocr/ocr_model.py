
import os
import sys
sys.path.append('.')
from easydict import EasyDict as edict
import numpy as np
from typing import List
import torch
from dataclasses import dataclass, asdict
from copy import deepcopy

from common_ml.tagging.models.frame_based import BatchFrameModel, FrameTag, FrameModel

from .ocr import OCR


@dataclass
class RuntimeConfig:
    w_thres: float = 0.8
    l_thres: float = 0.8

class OCRModel(FrameModel):
    def __init__(self, model_input_path: str, runtime_config: RuntimeConfig):

        self.model_input_path = model_input_path
        self.config = runtime_config
        
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
    
    """
    def tag_frames(self, imgs: np.ndarray) -> List[List[FrameTag]]:
        res = []
        ocr_batch = 10
        for i in range(0, len(imgs), ocr_batch):
            ocr_out = self.ocr_inference.inference(
                imgs[i:i+ocr_batch], self.args.word_batch_size, self.config.w_thres, self.config.l_thres)
            for img in ocr_out:
                res.append([])
                for tag in img['ocr']['tags']:
                    box = tag["box"]
                    # limit to only x1, x2, y1, y2
                    box = {"x1": box["x1"], "x2": box["x2"], "y1": box["y1"], "y2": box["y2"]}
                    text = tag["text"]
                    res[-1].append(FrameTag(tag=text, box=box))

        return res
    """

    def tag_frame(self, img: np.ndarray) -> List[FrameTag]:
        tags = self.ocr_inference.inference([img], self.args.word_batch_size, self.config.w_thres, self.config.l_thres)
        res = []
        for tag in tags[0]['ocr']['tags']:
            box = tag["box"]
            # limit to only x1, x2, y1, y2
            box = {"x1": box["x1"], "x2": box["x2"], "y1": box["y1"], "y2": box["y2"]}
            text = tag["text"]
            res[-1].append(FrameTag(tag=text, box=box))
        return res