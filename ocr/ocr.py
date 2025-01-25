
import math
import string
from collections import OrderedDict
import sys
sys.path.append('.')
from loguru import logger

import numpy as np
import cv2
import torch
from torch.nn import DataParallel
import torchvision.transforms as transforms

from .imageproc import normalizeMeanVariance
from .model import CRAFT, RefineNet, STRModel
from .boxproc import getDetBoxes, adjustResultCoordinates, getWordSequences
from .modules import AttnLabelConverter, CTCLabelConverter

def copyStateDict(state_dict):
    """
    CRAFT and LinkRefiner model state dict.
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


# def save_score_maps(result_dir, image_id, region, affinity, link=None):
#     """
#     Save CRAFT (and LinkRefiner) outputs.
#     """
#     np.savetxt(os.path.join(result_dir, image_id + '_region.txt'), region)
#     np.savetxt(os.path.join(result_dir, image_id + '_affinity.txt'), affinity)
#     if link is not None:
#         np.savetxt(os.path.join(result_dir, image_id + '_link.txt'), link)


class STRModelArgs:

    input_channel = None
    output_channel = None
    imgW = None
    imgH = None
    Transformation = None
    num_fiducial = None
    device = None
    FeatureExtraction = None
    SequenceModeling = None
    hidden_size = None
    Prediction = None
    num_class = None
    batch_max_length = None


class ResizeNormalize:

    def __init__(self, w, h, interpolation=cv2.INTER_CUBIC):
        self.w, self.h = w, h
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = cv2.resize(img, (self.w, self.h), interpolation=self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class ResizeNormalizePAD(object):

    def __init__(self, w, h, interpolation=cv2.INTER_CUBIC, PAD_type='right'):
        # w / h is the aspect ratio limit; if larger resize without pad
        self.w = w
        self.h = h
        # self.half_width = math.floor(w / 2)
        self.interpolation = interpolation
        self.PAD_type = PAD_type
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # Resize
        h, w, _ = img.shape
        resize_w = max(math.ceil(self.h / h * w), self.w)
        resized = cv2.resize(img, (resize_w, self.h), interpolation=self.interpolation)

        # Pad
        resized = self.toTensor(resized)
        resized.sub_(0.5).div_(0.5)
        c, h, w = resized.size()
        Pad_img = torch.FloatTensor(c, self.h, self.max_w).fill_(0)
        Pad_img[:, :, :w] = resized  # right pad
        if self.w != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.w - w)
        return Pad_img


class OCR:
    def __init__(self, device, craft_model, refiner_model, rec_model,
                 text_threshold=0.7, region_threshold=0.4, affinity_threshold=0.4, link_threshold=0.4,
                 mag_ratio=1.5, mag_limit=1280, word_poly=True, line_poly=True,
                 insensitive=False, keep_ratio_and_pad=False, word_rgb=False,
                 word_image_width=100, word_image_height=32,
                 Transformation='TPS', n_fiducial=20,
                 FeatureExtraction='ResNet', word_in_channels=1, feature_extraction_channels=512,
                 SequenceModeling='BiLSTM', hidden_size=256,
                 Prediction='Attn', max_text_len=25):
        self.device = device

        # Detection settings
        self.text_threshold = text_threshold
        self.region_threshold = region_threshold
        self.affinity_threshold = affinity_threshold
        self.link_threshold = link_threshold
        self.mag_ratio = mag_ratio
        self.mag_limit = mag_limit
        self.word_poly = word_poly
        self.line_poly = line_poly

        # Recognition settings
        self.word_image_height = word_image_height
        self.word_image_width = word_image_width
        self.keep_ratio_and_pad = keep_ratio_and_pad
        self.word_in_channels = word_in_channels
        self.word_rgb = word_rgb
        self.Prediction = Prediction  # CTC | Attn

        # Models
        logger.info('Initializing detection nets...')
        self.det_net = CRAFT()
        self.det_net.load_state_dict(copyStateDict(torch.load(craft_model, map_location=self.device)))
        self.det_net = DataParallel(self.det_net).to(self.device)
        self.det_net.eval()
        #logger.info('CRAFT on CUDA?', next(self.det_net.parameters()).is_cuda)

        self.refine_net = RefineNet()
        self.refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location=self.device)))
        self.refine_net = DataParallel(self.refine_net).to(self.device)
        self.refine_net.eval()
        #logger.info('LinkRefiner on CUDA?', next(self.refine_net.parameters()).is_cuda)

        logger.info('Initializing recognition net...')
        character = '0123456789abcdefghijklmnopqrstuvwxyz' if insensitive else string.printable[:-6]
        self.converter = CTCLabelConverter(character) if 'CTC' in Prediction else AttnLabelConverter(character)

        str_args = STRModelArgs()
        str_args.input_channel = self.word_in_channels
        str_args.output_channel = feature_extraction_channels
        str_args.imgW = self.word_image_width
        str_args.imgH = self.word_image_height
        str_args.Transformation = Transformation
        str_args.num_fiducial = n_fiducial
        str_args.device = self.device
        str_args.FeatureExtraction = FeatureExtraction
        str_args.SequenceModeling = SequenceModeling
        str_args.hidden_size = hidden_size
        str_args.Prediction = Prediction
        str_args.num_class = len(self.converter.character)
        str_args.batch_max_length = max_text_len

        self.rec_net = STRModel(str_args)
        self.rec_net = DataParallel(self.rec_net).to(self.device)
        self.rec_net.load_state_dict(torch.load(rec_model, map_location=self.device))
        self.rec_net.eval()
        #logger.info('STR on CUDA?', next(self.rec_net.parameters()).is_cuda)

    def inference(self, batch, word_batch_size=256, word_threshold=0.0, line_threshold=0.0):
        # Detection
        images, raw_hw, heatmap_hw, ratios = self.__image_preproc(batch)
        images = images.to(self.device)
        with torch.no_grad():
            y, feature = self.det_net(images)
            region = y[:, :, :, 0].cpu().numpy()
            affinity = y[:, :, :, 1].cpu().numpy()
            y = self.refine_net(y, feature)
            link = y[:, :, :, 0].cpu().numpy()

        detections = []
        boxes_list = []  # a list of (word_index, word_quads) for each image
        for k, ((rawh, raww), (hh, hw), ratio) in enumerate(zip(raw_hw, heatmap_hw, ratios)):
            R, A, L = region[k, :hh, :hw], affinity[k, :hh, :hw], link[k, :hh, :hw]

            word_quads, word_polys, word_labels, word_map = getDetBoxes(
                R, A, self.region_threshold, self.affinity_threshold, self.text_threshold, self.word_poly)
            word_quads = adjustResultCoordinates(word_quads, ratio, ratio)
            word_polys = adjustResultCoordinates(word_polys, ratio, ratio)

            line_quads, line_polys, line_labels, line_map = getDetBoxes(
                R, L, self.region_threshold, self.link_threshold, self.text_threshold, self.line_poly)
            line_quads = adjustResultCoordinates(line_quads, ratio, ratio)
            line_polys = adjustResultCoordinates(line_polys, ratio, ratio)
            word_seqs = getWordSequences(word_map, word_labels, line_map, line_labels)

            detections.append((zip(word_quads, word_polys, word_labels),
                               zip(line_quads, line_polys, line_labels, word_seqs),
                               rawh, raww))
            boxes_list.append(zip(word_labels, word_quads))

        # Recognition
        images, sample_indices, word_indices = self.__word_preproc(batch, boxes_list)
        if images is None:
            return self.__formatter(detections, dict())
        n_batches = (images.shape[0] - 1) // word_batch_size + 1
        transcripts = dict()
        for k in range(n_batches):
            I, J = k * word_batch_size, (k + 1) * word_batch_size
            imgs = images[I:J, ...].to(self.device)
            B = imgs.shape[0]
            length_for_pred = [None] * B
            text_for_pred = None

            if 'CTC' in self.Prediction:
                with torch.no_grad():
                    preds = self.rec_net(imgs, text_for_pred, is_train=False).softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * B)
                    preds_score, preds_index = preds.permute(1, 0, 2).max(2)
                    preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                    preds_str = self.converter.decode(preds_index, preds_size)
            else:
                with torch.no_grad():
                    preds = self.rec_net(imgs, text_for_pred, is_train=False).softmax(2)
                preds_score, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            # TODO word confidence
            preds_score = preds_score.mean(1)
            for si, wi, pred, conf in zip(sample_indices[I:J], word_indices[I:J], preds_str, preds_score):
                if 'Attn' in self.Prediction:
                    pred = pred[:pred.find('[s]')]  # prune after EOS token
                transcripts[(si, wi)] = (pred, conf.item())

        return self.__formatter(detections, transcripts, word_threshold, line_threshold)

    def __image_preproc(self, batch):
        target_h, target_w = 0, 0
        for image in batch:
            h, w, c = image.shape
            target_h, target_w = max(self.mag_ratio * h, target_h), max(self.mag_ratio * w, target_w)
        target_h, target_w = min(target_h, self.mag_limit), min(target_w, self.mag_limit)
        target_h, target_w = int(target_h), int(target_w)

        target_h32 = target_h + (32 - target_h % 32) if target_h % 32 != 0 else target_h
        target_w32 = target_w + (32 - target_w % 32) if target_w % 32 != 0 else target_w
        resized_images, raw_hw, heatmap_hw, ratios = [], [], [], []
        for image in batch:
            h, w, c = image.shape
            ratio = min(target_h / h, target_w / w)  # to adjust bounding box coordinates
            th, tw = int(h * ratio), int(w * ratio)

            # make canvas and paste resized image
            canvas = np.zeros((target_h32, target_w32, c), dtype=np.float32)
            resized = cv2.resize(image, (tw, th), interpolation=cv2.INTER_CUBIC)  #TODO interpolation
            canvas[0:th, 0:tw, :] = resized

            canvas = normalizeMeanVariance(canvas)  # H, W, C
            canvas = torch.from_numpy(canvas).permute(2, 0, 1)  # C, H, W
            resized_images.append(canvas)
            heatmap_hw.append((th // 2, tw // 2))
            raw_hw.append((h, w))
            ratios.append(1 / ratio)

        resized_images = torch.cat([x.unsqueeze(0) for x in resized_images], 0)
        return resized_images, raw_hw, heatmap_hw, ratios

    def __word_preproc(self, batch, boxes_list):
        word_images, sample_indices, word_indices = [], [], []
        H, W = self.word_image_height, self.word_image_width
        transform = ResizeNormalizePAD(W, H) if self.keep_ratio_and_pad else ResizeNormalize(W, H)

        for k, (image, boxes) in enumerate(zip(batch, boxes_list)):
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if self.word_rgb else cv2.COLOR_BGR2GRAY)
            for word_index, quad in boxes:
                sample_indices.append(k)
                word_indices.append(word_index)

                M = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
                word_image = cv2.warpPerspective(img, M, (W, H))
                word_images.append(transform(word_image))
        word_images = torch.cat([x.unsqueeze(0) for x in word_images]) if len(word_images) else None
        return word_images, sample_indices, word_indices

    def __formatter(self, detections, transcripts, word_threshold=0.0, line_threshold=0.0):
        results = dict()
        for si, (words, lines, rawh, raww) in enumerate(detections):
            wordres = []
            for k, (quad, poly, word_index) in enumerate(words):
                transcript, conf = transcripts[(si, word_index)]
                if conf <= word_threshold: continue
                wordres.append({
                    'text': transcript,
                    'confidence': round(conf,4),
                    'box': self.__box_formatter(quad, rawh, raww),
                    'polygon': self.__box_formatter(poly, rawh, raww),
                })

            lineres = []
            for k, (quad, poly, line_index, word_indices) in enumerate(lines):
                transcript, conf = [], 0
                for wi in word_indices:
                    t, c = transcripts[(si, wi)]
                    transcript.append(t)
                    conf = max(conf, c)  # TODO line confidence
                if conf <= line_threshold: continue
                transcript = ' '.join(transcript)
                lineres.append({
                    'text': transcript,
                    'confidence': round(conf,4),
                    'box': self.__box_formatter(quad, rawh, raww),
                    'polygon': self.__box_formatter(poly, rawh, raww),
                })

            results[si] = {'ocr': {'tags': lineres, 'words': wordres}}

        return results

    def __box_formatter(self, box, rawh, raww):
        x = box[:, 0] / raww
        y = box[:, 1] / rawh
        res = dict()
        for k in range(box.shape[0]):
            res[f'x{k + 1}'] = round(x[k].item(),4)
            res[f'y{k + 1}'] = round(y[k].item(),4)
        return res

