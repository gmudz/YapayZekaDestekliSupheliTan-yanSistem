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
        self.last_bag_detections: List[Tuple[int, int, int, int, float, str]] = []

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
            # COCO classes: 0=person, 24=backpack, 26=handbag, 28=suitcase
            self.model.classes = [0, 24, 26, 28]
            self.model.amp = False

            stride = int(getattr(self.model, "stride", 32))
            self.model_size = check_img_size((input_size, input_size), s=stride)
            self.model.warmup(imgsz=(1, 3, *self.model_size))
            self.model.to(self.device)
            self.model.eval()

            self._letterbox = letterbox
            self._nms = non_max_suppression
            self.backend = "yolov5"
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            raise RuntimeError(f"Error initializing YOLOv5 PersonDetector: {e}") from e

    def _detect_closeup_webcam_person(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """Fallback for close-up laptop webcam shots where mainly the user's face/head is visible."""
        orig_h, orig_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        faces = self._face_cascade.detectMultiScale(
            small, scaleFactor=1.15, minNeighbors=4, minSize=(28, 28)
        )
        results: List[Tuple[int, int, int, int, float]] = []
        for fx, fy, fw, fh in faces:
            fx, fy, fw, fh = int(fx * 2), int(fy * 2), int(fw * 2), int(fh * 2)
            # Expand face box to upper-body person box
            x1 = max(0, int(fx - fw * 0.6))
            y1 = max(0, int(fy - fh * 0.25))
            x2 = min(orig_w - 1, int(fx + fw * 1.6))
            y2 = min(orig_h - 1, int(fy + fh * 2.6))
            results.append((x1, y1, x2, y2, 0.88))
        return results

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
                    classes=[0, 24, 26, 28],
                    max_det=self.model.max_det,
                )

            results: List[Tuple[int, int, int, int, float]] = []
            bag_results: List[Tuple[int, int, int, int, float, str]] = []
            if not nms_out or nms_out[0] is None or len(nms_out[0]) == 0:
                results = self._detect_closeup_webcam_person(frame)
                self.last_detections_with_conf = results
                self.last_bag_detections = []
                return results

            boxes = nms_out[0].detach().cpu().numpy()
            boxes[:, :4] = scale_boxes_from_letterbox(
                boxes[:, :4], (orig_h, orig_w), ratio, (dw, dh)
            )

            bag_class_names = {24: "Backpack", 26: "Handbag", 28: "Suitcase"}
            for box in boxes:
                if len(box) >= 6:
                    xmin, ymin, xmax, ymax, conf, cls = box[:6]
                    xmin_i, ymin_i, xmax_i, ymax_i = (
                        int(round(xmin)),
                        int(round(ymin)),
                        int(round(xmax)),
                        int(round(ymax)),
                    )
                    cls_i = int(cls)
                    if (
                        cls_i == 0
                        and float(conf) >= self.model.conf
                        and xmax_i - xmin_i >= 12
                        and ymax_i - ymin_i >= 20
                    ):
                        results.append(
                            (xmin_i, ymin_i, xmax_i, ymax_i, float(conf))
                        )
                    elif cls_i in bag_class_names and float(conf) >= 0.35:
                        bag_results.append(
                            (xmin_i, ymin_i, xmax_i, ymax_i, float(conf), bag_class_names[cls_i])
                        )

            if not results:
                results = self._detect_closeup_webcam_person(frame)

            self.last_detections_with_conf = results
            self.last_bag_detections = bag_results
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
