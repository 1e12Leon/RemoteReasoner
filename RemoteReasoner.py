
"""
RemoteReasoner: A unified workflow for geospatial reasoning tasks.

This module provides the main RemoteReasoner class for performing various
geospatial reasoning tasks including region reasoning, pixel-level reasoning,
contour reasoning, VQA, and image captioning.

Copyright 2025 RemoteReasoner Authors
Licensed under the Apache License, Version 2.0
"""

import os
import re
import ast
import argparse
import numpy as np
from PIL import Image
import cv2
from loguru import logger
from swift.llm import InferEngine, InferRequest, VllmEngine, PtEngine, RequestConfig, get_template, \
    safe_snapshot_download, \
    BaseArguments
from swift.tuners import Swift
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from types import SimpleNamespace
# Import SAM mask generator
from sam2.sam_mask_generator import SAMMaskGenerator

# Prompt template
prompt = (
    "You are an object localization assistant. Your task is to locate the target object(s) in the image "
    "based on the provided text description and output bounding box coordinates. "
    "Please follow these requirements:\n"
    "1. Text comprehension: Carefully analyze the text description to identify the target object.\n"
    "2. Output format:\n"
    "   - If one object exists: Output the bounding box coordinates in the format: [[x1,y1,x2,y2]]\n"
    "   - If no matching object is found: Output an empty list []\n"
    "   - If multi-objects exist: Output the bounding box coordinates in the format: [[x1,y1,x2,y2],...,[x1,y1,x2,y2]]\n"
    "3. Coordinate requirements:\n"
    "   - Bounding box coordinates should be arranged as [x_min, y_min, x_max, y_max]\n"
    "   - Coordinate values must be integers representing pixel positions\n"
    "4. Output specifications:\n"
    "   - Output the thinking process in <think> </think> and \n final answer (a list of bounding boxes) in <answer> </answer> tags."
    "Perform the grounding task for the following input:"
)

prompt_QA = (
    "You are an useful assistant. Your task is to answer the question." 
    "Please follow these requirements:\n"
    "1. Text comprehension: Carefully analyze the text description to answer the question.\n"
    "2. Output specifications:\n"
    "   - Output the thinking process in <think> </think> and \n final answer in <answer> </answer> tags."
    "Here is the question:"
)

prompt_caption = (
    "You are an useful assistant. Please discribe the image." 
)

class RemoteReasoner:
    """
    RemoteReasoner class for unified geospatial reasoning tasks.
    
    This class integrates a multi-modal large language model (MLLM) for interpreting
    user instructions and localizing targets, together with SAM for multi-granularity
    tasks including object-, region-, and pixel-level reasoning.
    
    Attributes:
        request_config: Configuration for inference requests
        engine: LLM inference engine (PtEngine)
        sam_generator: SAM mask generator for pixel-level tasks
    """
    
    def __init__(self, args, device=0):
        """
        Initialize inference engine and SAM mask generator.
        
        Args:
            args: Argument object containing model_path and other configurations
            device: GPU device ID to use (default: 0)
        """
        self.request_config = RequestConfig()
        self.engine = self._init_llm(args, device)
        self.sam_generator = self._init_sam()
        logger.info("RemoteReasoner initialized with LLM and SAM models")
    
    def _init_llm(self, args, device):
        """
        Initialize LLM inference engine.
        
        Args:
            args: Arguments containing model_path
            device: GPU device ID
            
        Returns:
            PtEngine: Initialized PyTorch inference engine
        """
        model_path = safe_snapshot_download(args.model_path)
        args_info = BaseArguments.from_pretrained(model_path)
        args_info.device_map = device
        model, processor = args_info.get_model_processor()

        template = args_info.get_template(processor)
        return PtEngine.from_model_template(model, template)
    
    def _init_sam(self):
        """
        Initialize SAM mask generator.
        
        Returns:
            SAMMaskGenerator: Initialized SAM mask generator
        """
        # Use relative path from current working directory
        sam_checkpoint = "sam2.1_hiera_tiny.pt"
        sam_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        return SAMMaskGenerator(sam_checkpoint, sam_config)
    
    def _parse_bboxes(self, answer_str):
        """
        Parse bounding box coordinates from model output.
        
        Validates and extracts bounding box coordinates from the model's answer string.
        Handles both single and multiple object cases.
        
        Args:
            answer_str: Answer string from model output containing bbox coordinates
            
        Returns:
            list: Parsed bbox list in format [[x1, y1, x2, y2], ...]. 
                  Returns empty list if parsing fails or format is invalid.
                  
        Examples:
            >>> _parse_bboxes("[[100, 200, 300, 400]]")
            [[100, 200, 300, 400]]
            >>> _parse_bboxes("[[10, 20, 30, 40], [50, 60, 70, 80]]")
            [[10, 20, 30, 40], [50, 60, 70, 80]]
        """
        try:
            # Try to parse model output
            bbox_list = ast.literal_eval(answer_str)
            
            # Validate format
            if not isinstance(bbox_list, list):
                return []
                
            # Handle single object case
            if len(bbox_list) > 0 and not isinstance(bbox_list[0], list):
                bbox_list = [bbox_list]
                
            # Filter invalid bboxes
            valid_bboxes = []
            for bbox in bbox_list:
                if isinstance(bbox, list) and len(bbox) == 4:
                    if all(isinstance(coord, (int, float)) for coord in bbox):
                        valid_bboxes.append(bbox)
            
            return valid_bboxes
        except Exception as e:
            logger.warning(f"Failed to parse bbox from: {answer_str}, Error: {e}")
            return []
        
    def mask2contour(self, mask):
        """
        Extract contours from a segmentation mask.
        
        Converts a binary segmentation mask into a contour representation by
        detecting edges and drawing them on a white background.

        Args:
            mask: PIL.Image object (single channel, values 0/255 or 0/1)

        Returns:
            PIL.Image: Contour image with black contours on white background
            
        Examples:
            >>> mask = Image.open("mask.png")
            >>> contour = reasoner.mask2contour(mask)
            >>> contour.save("contour.png")
        """

        # Convert mask to numpy array
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            # If mask is RGB, convert to grayscale
            mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)

        _, binary = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)
        contour_result = np.ones((binary.shape[0], binary.shape[1], 3), dtype=np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_result, contours, -1, (0, 0, 0), 3)

        contour_img = Image.fromarray(contour_result)

        return contour_img
    
    def Region_reasoning(self, image, question):      
        """
        Perform region-level reasoning to generate bounding boxes.
        
        This method takes an image and a query, then uses the LLM to identify
        target objects and output their bounding box coordinates.
        
        Args:
            image: Either a file path (str) to the image or a PIL.Image object
            question: Query or referring expression describing the target object(s)
            
        Returns:
            tuple: (think, answer) where:
                - think (str): The reasoning process/thought chain
                - answer (str): The final answer containing bbox coordinates
                
        Raises:
            FileNotFoundError: If image path does not exist
            TypeError: If image is neither a file path nor PIL.Image object
            
        Examples:
            >>> think, answer = reasoner.Region_reasoning("path/to/image.jpg", "find the car")
            >>> print(answer)  # "[[100, 200, 300, 400]]"
        """
        if isinstance(image, str):
            # If path, ensure file exists
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be either a file path or PIL.Image object")
        
        messages = [{
            "role": "user",
            "content": f"{prompt}\n<image>\nText description: {question}"
        }]
        
        resp_list = self.engine.infer([{
            "images": image,
            "messages": messages
        }], self.request_config)
        
        response = resp_list[0]
        content = response.choices[0].message.content
        
        match = re.match(
            r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$',
            content, 
            re.DOTALL | re.MULTILINE
        )
        if match:
            return match.group(1), match.group(2)  # think, answer
        return "", content  # Return original content if parsing fails
    
    def Pixel_reasoning(self, image, question):
        """
        Perform pixel-level reasoning to generate bounding boxes and segmentation masks.
        
        This method first performs region reasoning to get bounding boxes, then uses
        SAM to convert them into pixel-level segmentation masks.
        
        Args:
            image: Either a file path (str) to the image or a PIL.Image object
            question: Query or referring expression describing the target object(s)
            
        Returns:
            tuple: (think, answer, mask) where:
                - think (str): The reasoning process/thought chain
                - answer (str): The bbox coordinates
                - mask (PIL.Image): Binary segmentation mask (0/255 values)
                
        Examples:
            >>> think, answer, mask = reasoner.Pixel_reasoning("path/to/image.jpg", "segment the building")
            >>> mask.save("output_mask.png")
        """
        think, answer = self.Region_reasoning(image, question)
 
        bboxes = self._parse_bboxes(answer)
       
        if isinstance(image, str):
            mask_array = self.sam_generator.generate_mask(image, bboxes)
        else:
            image_np = np.array(image)
            mask_array = self.sam_generator.generate_mask(image_np, bboxes)
        mask_img = Image.fromarray(mask_array * 255)
        
        return think, answer, mask_img
    
    def Contour_reasoning(self, image, question):
        """
        Perform contour-level reasoning to generate bounding boxes and contours.
        
        This method performs pixel-level reasoning to get masks, then extracts
        contours from the masks for boundary representation.

        Args:
            image: Either a file path (str) to the image or a PIL.Image object
            question: Query or referring expression describing the target object(s)
            
        Returns:
            tuple: (think, answer, contour) where:
                - think (str): The reasoning process/thought chain
                - answer (str): The bbox coordinates
                - contour (PIL.Image): Contour image with black edges on white background
                
        Examples:
            >>> think, answer, contour = reasoner.Contour_reasoning("image.jpg", "outline the road")
            >>> contour.save("road_contour.png")
        """
        think, answer, mask = self.Pixel_reasoning(image, question)

        contour = self.mask2contour(mask)

        return think, answer, contour

    def VQA(self, image, question):
        """
        Perform Visual Question Answering on the given image.
        
        Args:
            image: Either a file path (str) to the image or a PIL.Image object
            question: Question about the image content
            
        Returns:
            tuple: (think, answer) where:
                - think (str): The reasoning process/thought chain
                - answer (str): The answer to the question
                
        Examples:
            >>> think, answer = reasoner.VQA("satellite.jpg", "How many buildings are visible?")
            >>> print(answer)  # "There are 5 buildings visible in the image."
        """
        messages = [{
            "role": "user",
            "content": f"{prompt_QA}\n<image>\nQuestion: {question}"
        }]
        
        resp_list = self.engine.infer([{
            "images": image,
            "messages": messages
        }], self.request_config)
        
        response = resp_list[0]
        content = response.choices[0].message.content
        
        match = re.match(
            r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$',
            content, 
            re.DOTALL | re.MULTILINE
        )
        if match:
            return match.group(1), match.group(2)  # think, answer
        return "", content
    
    def Image_captioning(self, image):
        """
        Generate a descriptive caption for the given image.
        
        Args:
            image: Either a file path (str) to the image or a PIL.Image object
            
        Returns:
            tuple: (think, answer) where:
                - think (str): The reasoning process/thought chain  
                - answer (str): The generated image caption
                
        Examples:
            >>> think, caption = reasoner.Image_captioning("aerial_view.jpg")
            >>> print(caption)  # "An aerial view of a coastal city with dense urban development."
        """      
        messages = [{
            "role": "user",
            "content": f"{prompt_caption}"
        }]
        
        resp_list = self.engine.infer([{
            "images": image,
            "messages": messages
        }], self.request_config)
        
        response = resp_list[0]
        content = response.choices[0].message.content
        
        match = re.match(
            r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$',
            content, 
            re.DOTALL | re.MULTILINE
        )
        if match:
            return match.group(1), match.group(2)  # think, answer
        return "", content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=False, default="checkpoints/RemoteReasoner-7B-merged-bf16",
                        help="Path to the model")
    args = parser.parse_args()
    
    logger.info("Initializing RemoteReasoner...")
    reasoner = RemoteReasoner(args, device=0)
    
    test_cases = [
        ("assets/demo.jpg", "Considering its role in biodiversity conservation, what distinct feature visible from above here serves as habitat and refuge amidst vast waters?"),
    ]
    
    for img_path, question in test_cases:
        logger.info(f"Processing image: {img_path}")
        

        think, answer = reasoner.Region_reasoning(img_path, question)
        logger.info(f"Think: {think}")
        logger.info(f"Answer: {answer}")
        print(f"Think: {think}")
        print(f"Answer: {answer}")
        # Pixel-level reasoning
        think, answer, mask = reasoner.Pixel_reasoning(img_path, question)
        # Save generated mask
        mask_path = os.path.splitext(img_path)[0] + "_mask.png"
        mask.save( mask_path)
        logger.info(f"Mask saved to: {mask_path}")
        print(f"Think: {think}")
        print(f"Answer: {answer}")

