"""Module for person detection using YOLOv5 (Edge Layer — TÜBİTAK 2209-A).

Provides the PersonDetector class that uses YOLOv5 (yolov5m.pt) for real-time
person detection in video streams with accurate letterbox coordinate rescaling,
automatic CUDA/CPU device selection, and optional FP16/TensorRT acceleration.
"""

import os
import cv2
import torch
import torchvision
import numpy as np
from typing import List, Tuple, Optional


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Calculate intersection over union (IoU) between two sets of boxes."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter + 1e-7)


def scale_boxes_from_letterbox(
    boxes: np.ndarray,
    orig_shape: Tuple[int, int],
    ratio: Tuple[float, float],
    pad: Tuple[float, float],
) -> np.ndarray:
    """Rescale bounding boxes from letterboxed image space back to original frame coordinates."""
    if len(boxes) == 0:
        return boxes
    scaled = boxes.copy().astype(np.float32)
    dw, dh = pad
    gain_w, gain_h = ratio
    scaled[:, [0, 2]] -= dw
    scaled[:, [1, 3]] -= dh
    scaled[:, [0, 2]] /= max(gain_w, 1e-6)
    scaled[:, [1, 3]] /= max(gain_h, 1e-6)
    h, w = orig_shape[:2]
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, w - 1)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, h - 1)
    return scaled


class PersonDetector:
    """Real-time person detector using YOLOv5 (Edge Layer).

    Args:
        model_name: Path to YOLOv5 weights file (default: 'yolov5m.pt')
        confidence_threshold: Minimum confidence threshold for person detections
        iou_threshold: IoU threshold for Non-Maximum Suppression (NMS)
        target_classes: Target classes to detect (default: ['person'])
        device: 'auto', 'cuda:0', or 'cpu'
        half_precision: Whether to use FP16 half precision on GPU
        input_size: Inference resolution (default: 416)
    """

    def __init__(
        self,
        model_name: str = "yolov5m.pt",
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
        device: str = "auto",
        half_precision: bool = False,
        input_size: int = 416,
    ):
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.target_classes = target_classes or ["person"]
        self.last_detections_with_conf: List[Tuple[int, int, int, int, float]] = []

        if device in ("auto", "", None):
            resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        else:
            resolved_device = device

        self.device_str = resolved_device
        self.half_precision = bool(half_precision and "cuda" in resolved_device)

        # Resolve model path relative to project root if needed
        if not os.path.exists(model_name):
            candidate = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                model_name,
            )
            if os.path.exists(candidate):
                model_name = candidate

        self.model_path = model_name

        try:
            from yolov5.models.common import DetectMultiBackend
            from yolov5.utils.general import non_max_suppression, check_img_size
            from yolov5.utils.augmentations import letterbox
            from yolov5.utils.torch_utils import select_device

            # Modern PyTorch (>=2.6) sets weights_only=True by default in torch.load,
            # while YOLOv5 checkpoints contain DetectionModel classes.
            _orig_torch_load = torch.load

            def _compat_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_torch_load(*args, **kwargs)

            torch.load = _compat_load
            try:
                self.device = select_device(resolved_device)
                self.model = DetectMultiBackend(
                    self.model_path,
                    device=self.device,
                    fp16=self.half_precision,
                )
            finally:
                torch.load = _orig_torch_load

            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model.max_det = 60
            self.model.classes = [0]  # COCO class 0 = person
            self.model.amp = False

            stride = int(getattr(self.model, "stride", 32))
            self.model_size = check_img_size((input_size, input_size), s=stride)
            self.model.warmup(imgsz=(1, 3, *self.model_size))
            self.model.to(self.device)
            self.model.eval()

            self._letterbox = letterbox
            self._nms = non_max_suppression
            self.backend = "yolov5"
        except Exception as e:
            raise RuntimeError(f"Error initializing YOLOv5 PersonDetector: {e}") from e

    def detect_with_conf(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """Detect people in a BGR frame and return [(xmin, ymin, xmax, ymax, confidence)]."""
        if frame is None or frame.size == 0:
            self.last_detections_with_conf = []
            return []

        try:
            orig_h, orig_w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lb_img, ratio, (dw, dh) = self._letterbox(
                img_rgb, self.model_size, auto=False
            )

            tensor = torch.from_numpy(lb_img).to(self.device)
            tensor = tensor.half() if self.half_precision else tensor.float()
            tensor /= 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()

            with torch.no_grad():
                pred = self.model(tensor)
                nms_out = self._nms(
                    pred,
                    conf_thres=self.model.conf,
                    iou_thres=self.model.iou,
                    classes=[0],
                    max_det=self.model.max_det,
                )

            results: List[Tuple[int, int, int, int, float]] = []
            if not nms_out or nms_out[0] is None or len(nms_out[0]) == 0:
                self.last_detections_with_conf = []
                return results

            boxes = nms_out[0].detach().cpu().numpy()
            boxes[:, :4] = scale_boxes_from_letterbox(
                boxes[:, :4], (orig_h, orig_w), ratio, (dw, dh)
            )

            for box in boxes:
                if len(box) >= 6:
                    xmin, ymin, xmax, ymax, conf, cls = box[:6]
                    xmin_i, ymin_i, xmax_i, ymax_i = (
                        int(round(xmin)),
                        int(round(ymin)),
                        int(round(xmax)),
                        int(round(ymax)),
                    )
                    if (
                        int(cls) == 0
                        and float(conf) >= self.model.conf
                        and xmax_i - xmin_i >= 12
                        and ymax_i - ymin_i >= 20
                    ):
                        results.append(
                            (xmin_i, ymin_i, xmax_i, ymax_i, float(conf))
                        )

            self.last_detections_with_conf = results
            return results
        except Exception as e:
            print(f"[PersonDetector] Inference warning: {e}")
            self.last_detections_with_conf = []
            return []

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect people and return [(xmin, ymin, xmax, ymax)] for backward compatibility."""
        dets = self.detect_with_conf(frame)
        return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _conf) in dets]

    def __del__(self):
        if hasattr(self, "model"):
            try:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
