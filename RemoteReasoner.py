
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
# Import SAM mask generator
from sam_mask_generator import SAMMaskGenerator

# Prompt template
prompt = (
    "You are an image segmentation assistant. Your task is to locate the target object in the image "
    "based on the provided text description and output its segmentation polygon coordinates. "
    "Please follow these requirements:\n"
    "1. Text comprehension: Carefully analyze the text description to identify the target object.\n"
    "2. Output format:\n"
    "   - If one object exists: Output the polygon coordinates in the format: [[x1,y1,x2,y2,...,xn,yn]]\n"
    "   - If no matching object is found: Output an empty list []\n"
    "   - If multi-objects exist: Output the polygon coordinates in the format: [[x1,y1,x2,y2,...,xn,yn],...,[x1,y1,x2,y2,...,xn,yn]]\n"
    "3. Coordinate requirements:\n"
    "   - Polygon coordinates should be arranged clockwise or counter-clockwise along the object boundary\n"
    "   - Coordinate values must be integers representing pixel positions\n"
    "   - The coordinate sequence must form a closed polygon (first and last points connect)\n"
    "4. Output specifications:\n"
    "   - Output the thinking process in <think> </think> and \n final answer (a list of ploygon) in <answer> </answer> tags."
    "Perform the segmentation task for the following input:"
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
    def __init__(self, args, device=0):
        """
        Initialize inference engine and SAM mask generator
        
        Args:
            args: Argument object containing model path
            device: GPU device ID to use
        """
        self.request_config = RequestConfig()
        self.engine = self._init_llm(args, device)
        self.sam_generator = self._init_sam()
        logger.info("RemoteReasoner initialized with LLM and SAM models")
    
    def _init_llm(self, args, device):
        """Initialize LLM inference engine"""
        adapter_path = safe_snapshot_download(args.lora_path)
        args_info = BaseArguments.from_pretrained(adapter_path)
        args_info.device_map = device
        model, processor = args_info.get_model_processor()
        model = Swift.from_pretrained(model, adapter_path)
        template = args_info.get_template(processor)
        return PtEngine.from_model_template(model, template)
    
    def _init_sam(self):
        """Initialize SAM mask generator"""
        sam_checkpoint = "/home/leon.yao/code/sam2/checkpoints/sam2.1_hiera_large.pt"
        sam_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        return SAMMaskGenerator(sam_checkpoint, sam_config)
    
    def _parse_bboxes(self, answer_str):
        """
        Parse bbox coordinates from model output
        
        Args:
            answer_str: answer string from model output
            
        Returns:
            Parsed bbox list [[x1, y1, x2, y2], ...]
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
        except:
            logger.warning(f"Failed to parse bbox from: {answer_str}")
            return []
        
    def mask2contour(self, mask):
        """
        Extract contour from segmentation mask.

        Args:
            mask: PIL.Image object (single channel, 0/255 or 0/1)

        Returns:
            contour_img: PIL.Image object with contours drawn
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
        region_reasoning: generate bbox
        
        Args:
            image: image path or PIL.Image Object
            question: query or referring expression
            
        Returns:
            (think, answer) tuple
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
        pixel-level reasoning: generate bbox and convert to mask

        Args:
            image: image path or PIL.Image Object
            question: query or referring expression
            
        Returns:
            (think, answer, mask) triplet
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
        contour reasoning: generate bbox and convert to contour

        Args:
            image: image path or PIL.Image Object
            question: query or referring expression
            
        Returns:
            (think, answer, contour) triplet
        """
        think, answer, mask = self.Pixel_reasoning(image, question)

        contour = self.mask2contour(mask)

        return think, answer, contour

    def VQA(self, image, question):
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
    parser.add_argument('--lora_path', type=str, required=True, 
                        help="Path to the LoRA adapter model")
    args = parser.parse_args()
    
    logger.info("Initializing RemoteReasoner...")
    reasoner = RemoteReasoner(args, device=0)
    
    test_cases = [
        ("/disk/deepdata/dataset/RemoteReason/test/images/1277.jpg", "How many ships in the image?"),
        ("/disk/deepdata/dataset/RemoteReason/test/images/1289.jpg", "What the color is the river?"),
        ("/disk/deepdata/dataset/RemoteReason/test/images/3742.jpg", "The venue in the image can be used for what sports?"),
        ("/disk/deepdata/dataset/RemoteReason/test/images/4214.jpg", "Does the image have a bridge?"),
        ("/disk/deepdata/dataset/RemoteReason/test/images/0451.jpg", "What type is this image rural or urban?"),
        ("/disk/deepdata/dataset/RemoteReason/test/images/1439.jpg", "How many red roof buildings in the image?"),
        ("/disk/deepdata/dataset/RemoteReason/test/images/2505.jpg", "What the shape is the island?"),
        ("/disk/deepdata/dataset/RemoteReason/test/images/0534.jpg", "Which sports can people in this scene can do?")
    ]
    
    for img_path, question in test_cases:
        logger.info(f"Processing image: {img_path}")
        
        # Pixel-level reasoning
        think, answer, mask = reasoner.Pixel_reasoning(img_path, question)
        
        logger.info(f"Think: {think}")
        logger.info(f"Answer: {answer}")
        
        # Save generated mask
        mask_path = os.path.splitext(img_path)[0] + "_mask.png"
        mask.save(os.path.join("/disk/deepdata/dataset/RemoteReason", mask_path))
        logger.info(f"Mask saved to: {mask_path}")

