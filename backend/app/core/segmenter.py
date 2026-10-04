"""
Lesion Segmenter — isolates the skin lesion from the background.
Uses DeepLabV3-MobileNetV3 for salient object detection and automatic cropping.
"""
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)

class Segmenter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Preprocessing for segmentation
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model()

    def _load_model(self):
        """Loads a pre-trained DeepLabV3 model for foreground isolation."""
        try:
            # We use MobileNetV3 backbone for speed on CPU
            self.model = models.segmentation.deeplabv3_mobilenet_v3_large(
                weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            )
            self.model.to(self.device).eval()
            logger.info("Segmentation model loaded (DeepLabV3-MobileNetV3)")
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            self.model = None

    def segment_and_crop(self, image_path: str, output_path: str = None) -> str:
        """
        Detects the lesion, generates a mask, and crops the image to the lesion area.
        Returns the path to the cropped image.
        """
        if self.model is None:
            logger.warning("Segmenter not ready, skipping segmentation.")
            return image_path, None

        try:
            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
                # DeepLabV3 output is (num_classes, H, W). We take the max across classes.
                # For general saliency, we can just look for the class with highest energy.
                output_predictions = output.argmax(0).cpu().numpy()
            
            # Create a binary mask (assuming lesion is the foreground)
            # COCO classes: 0 is background. Any non-zero is potentially our lesion.
            mask = (output_predictions > 0).astype(np.uint8) * 255
            
            # Resize mask back to original image size
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Find contours to get the bounding box of the lesion
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.info("No clear lesion detected by segmenter, using original image.")
                return image_path, None
                
            # Get the largest contour (the lesion)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding (20%) to ensure we don't cut off the edges of the lesion
            pad_w = int(w * 0.2)
            pad_h = int(h * 0.2)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(orig_w, x + w + pad_w)
            y2 = min(orig_h, y + h + pad_h)
            
            # Crop
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Save the cropped image
            if output_path is None:
                output_path = str(Path(image_path).parent / f"cropped_{Path(image_path).name}")
                
            cropped_image.save(output_path)
            
            # Also save the mask for visualization if needed
            mask_path = str(Path(image_path).parent / f"mask_{Path(image_path).name}")
            cv2.imwrite(mask_path, mask)
            
            logger.info(f"Lesion segmented and cropped: {output_path}")
            return output_path, mask_path

        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            return image_path, None

segmenter = Segmenter()
