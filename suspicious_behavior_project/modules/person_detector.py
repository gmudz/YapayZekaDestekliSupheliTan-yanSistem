"""Module for person detection using YOLOv5.

This module provides a PersonDetector class that uses YOLOv5 for detecting
persons in video frames. It's designed to be used as part of a larger
video analytics pipeline.
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox

# Add NMS function
def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """Calculate intersection over union between two sets of boxes.
    
    Args:
        boxes1 (torch.Tensor): First set of boxes
        boxes2 (torch.Tensor): Second set of boxes
        
    Returns:
        torch.Tensor: Matrix of IoU values
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """Performs Non-Maximum Suppression (NMS) on inference results"""
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32
    
    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = prediction[:, :4]
    
    # Detections matrix nx6 (xyxy, conf, cls)
    conf = prediction[:, 4]
    cls = prediction[:, 5:]
    x = prediction[conf.view(-1) > conf_thres]
    
    # If none remain process next image
    if not x.shape[0]:
        return [torch.zeros(0, 6)]
    
    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
    
    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = x[:, :4]
    conf, j = x[:, 5:].max(1, keepdim=True)
    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
    
    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return [torch.zeros(0, 6)]
    
    # Batched NMS
    c = x[:, 5:6] * (0 if x[:, 5:6].max() < 10 else 1000)  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    
    output = [x[i]]
    return output


class PersonDetector:
    """Class for detecting people using YOLOv5 model.
    
    Args:
        model_path (str): Path to the YOLOv5 model weights
        confidence_threshold (float): Minimum confidence threshold for detections
        target_classes (List[str]): List of class names to detect
        device (str): Device to run inference on ('cuda' or 'cpu')
        half_precision (bool): Whether to use half precision (FP16) for inference
    """
    
    def __init__(self, 
                model_name: str = 'yolov5m.pt',
                confidence_threshold: float = 0.45,
                target_classes: List[str] = ['person'],
                device: str = 'cpu',
                half_precision: bool = False):
        """Initialize the person detector.
        
        Args:
            model_name (str): Name of the YOLOv5 model to use (e.g., 'yolov5m', 'yolov5l')
            confidence_threshold (float): Minimum confidence threshold for detections
            target_classes (List[str]): List of class names to detect
            device (str): Device to run inference on ('cuda' or 'cpu')
            half_precision (bool): Whether to use half precision (FP16) for inference
        """
        try:
            # Load model directly from yolov5 repository
            from yolov5.models.common import DetectMultiBackend
            from yolov5.utils.general import non_max_suppression, check_img_size
            from yolov5.utils.augmentations import letterbox
            from yolov5.utils.torch_utils import select_device
            
            # Set device
            self.device = select_device(device)
            
            # Load model
            self.model = DetectMultiBackend(model_name, device=self.device)
            
            # Set model parameters for better performance
            self.model.conf = confidence_threshold  # Set confidence threshold
            self.model.iou = 0.45  # Set IoU threshold
            self.model.max_det = 50  # Reduce max detections to improve speed
            self.model.classes = [0]  # Only detect people (class 0)
            self.model.amp = False  # Disable automatic mixed precision
            self.model_size = (320, 320)  # Reduce model size for faster inference
            self.model_size = check_img_size(self.model_size, s=self.model.stride)
            
            # Warm up model
            self.model.warmup(imgsz=(1, 3, *self.model_size))
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            
            # Cache model parameters
            self.model_size_cache = self.model_size
            self.device_cache = self.device
            
            # Store non_max_suppression for later use
            self.non_max_suppression = non_max_suppression
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect people in a single frame.
        
        Args:
            frame (np.ndarray): Input image frame in BGR format
            
        Returns:
            List of detections in format [(xmin, ymin, xmax, ymax)]
        """
        try:
            # Preprocess frame with caching
            if frame.shape[:2] != self.model_size_cache:
                # Only resize if frame size has changed
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = letterbox(img, self.model_size_cache, auto=False)[0]
                img = torch.from_numpy(img).to(self.device)
                img = img.float() / 255.0
                img = img.permute(2, 0, 1).unsqueeze(0)
                img = img.contiguous()
            else:
                # Reuse previous preprocessing if frame size is the same
                img = torch.from_numpy(frame).to(self.device)
                img = img.float() / 255.0
                img = img.permute(2, 0, 1).unsqueeze(0)
                img = img.contiguous()
            
            # Run inference
            with torch.no_grad():
                pred = self.model(img)
                pred = self.non_max_suppression(pred, conf_thres=self.model.conf, iou_thres=self.model.iou)[0]
            
            # Process results
            detections_list = []
            if pred is None or len(pred) == 0:
                print("No detections found")
                return detections_list
            
            print(f"Raw prediction shape: {pred.shape}")
            boxes = pred.cpu().numpy()
            print(f"Boxes shape: {boxes.shape}")
            print(f"First box: {boxes[0] if len(boxes) > 0 else 'No boxes'}")
            
            # Validate boxes before processing
            valid_boxes = []
            for box in boxes:
                if len(box) == 6:  # Ensure we have all 6 values (bbox + conf + class)
                    # Scale boxes back to original frame size
                    xmin = int(box[0] * frame.shape[1] / self.model_size[0])
                    ymin = int(box[1] * frame.shape[0] / self.model_size[1])
                    xmax = int(box[2] * frame.shape[1] / self.model_size[0])
                    ymax = int(box[3] * frame.shape[0] / self.model_size[1])
                    
                    # Validate box coordinates
                    if (xmin >= 0 and ymin >= 0 and xmax <= frame.shape[1] and 
                        ymax <= frame.shape[0] and xmax > xmin and ymax > ymin):
                        valid_boxes.append((xmin, ymin, xmax, ymax, box[4], box[5]))
            
            # Process valid boxes
            for box in valid_boxes:
                xmin, ymin, xmax, ymax, conf, cls = box
                
                # Only consider person class (class 0) with minimum confidence
                if cls == 0 and conf >= self.model.conf:
                    # Add debug logging
                    print(f"Detected person: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
                    print(f"Confidence: {conf:.2f}")
                    detections_list.append((xmin, ymin, xmax, ymax))
            
            print(f"Total detections: {len(detections_list)}")
            print(f"Detections: {detections_list}")
            return detections_list
        except Exception as e:
            print(f"Detection error: {str(e)}")
            print(f"Raw prediction: {pred if 'pred' in locals() else 'No prediction'}")
            return []

    def __del__(self):
        """Cleanup resources when object is destroyed."""
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
