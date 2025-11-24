# sam_mask_generator.py
import os
import ast
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# Initialize SAM model
class SAMMaskGenerator:
    def __init__(self, checkpoint_path, model_config_path):
        """
        Initialize SAM model
        :param checkpoint_path: Path to SAM model checkpoint
        :param model_config_path: Path to SAM model config file
        """
        self.checkpoint = checkpoint_path
        self.model_cfg = model_config_path
        self.predictor = SAM2ImagePredictor(build_sam2(model_config_path, checkpoint_path))
        print(f"SAM model loaded from {checkpoint_path}")
    
    def _process_image(self, image_input):
        """
        Process image input, support file path or numpy array
        :param image_input: Image path or numpy array
        :return: Standardized image numpy array
        """
        if isinstance(image_input, str):
            # Input is file path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = np.array(Image.open(image_input).convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            # Input is numpy array
            if image_input.ndim != 3 or image_input.shape[2] != 3:
                raise ValueError("Input image array must be in HWC format with 3 channels (RGB)")
            image = image_input
        else:
            raise TypeError("image_input must be either a file path (str) or numpy array")
        
        return image
    
    def _process_bboxes(self, bboxes):
        """
        Process bbox input, unify format
        :param bboxes: List of bounding boxes
        :return: Normalized bbox list
        """
        if not bboxes:
            return []
        
        # Unify format: ensure all bboxes are list type
        if not isinstance(bboxes, list):
            raise TypeError("bboxes must be a list")
        
        # Check format of each bbox
        normalized_bboxes = []
        for bbox in bboxes:
            if isinstance(bbox, str):
                # If bbox is string format (e.g. "[x1,y1,x2,y2]"), try to parse
                try:
                    bbox = ast.literal_eval(bbox)
                except:
                    raise ValueError(f"Invalid bbox string format: {bbox}")
            
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"Each bbox must be a list/tuple of 4 coordinates, got {bbox}")
            
            normalized_bboxes.append([float(coord) for coord in bbox])
        
        return normalized_bboxes
    
    def generate_mask(self, image_input, bboxes):
        """
        Generate mask based on input image and bboxes
        :param image_input: Image path or numpy array (H, W, 3)
        :param bboxes: List of bounding boxes, each as [x1, y1, x2, y2]
        :return: Binary mask numpy array (H, W), values are 0 or 1
        """
        # Process input image
        image = self._process_image(image_input)
        height, width = image.shape[:2]
        
        # Process bbox input
        normalized_bboxes = self._process_bboxes(bboxes)
        
        # Default: create all-zero mask
        mask_pred = np.zeros((height, width), dtype=np.uint8)
        
        # If no bbox, return empty mask
        if not normalized_bboxes:
            return mask_pred
        
        # Prepare input data
        input_boxes = np.array(normalized_bboxes)
        input_points = []
        input_labels = []
        
        for bbox in normalized_bboxes:
            # Calculate center point coordinates (x, y)
            x_center = int((bbox[0] + bbox[2]) / 2)
            y_center = int((bbox[1] + bbox[3]) / 2)
            input_points.append([x_center, y_center])
            input_labels.append(1)  # Foreground point label
        
        input_points = np.array(input_points)
        input_labels = np.array(input_labels)
        
        # Use SAM for prediction
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_boxes,
                multimask_output=False  # Return best mask
            )
            
            # Merge all predicted masks (logical OR)
            mask_pred = np.any(masks, axis=0).astype(np.uint8)
        
        return mask_pred

# Example usage
if __name__ == "__main__":
    # Initialize model
    checkpoint = "/home/dishimin/yaoliang/RemoteReasoner/sam2.1_hiera_large.pt"
    model_cfg = "/home/dishimin/yaoliang/RemoteReasoner/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    mask_generator = SAMMaskGenerator(checkpoint, model_cfg)
    
    # Example input
    image_path = "/path/to/your/image.jpg"
    bboxes = [
        [100, 100, 200, 200],  # [x1, y1, x2, y2]
        [300, 150, 400, 250]
    ]
    
    # Generate mask
    mask = mask_generator.generate_mask(image_path, bboxes)
    
    # Save result
    mask_img = Image.fromarray(mask * 255)
    mask_img.save("output_mask.png")
    print("Mask saved as output_mask.png")