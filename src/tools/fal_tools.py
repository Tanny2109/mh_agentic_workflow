"""Custom fal.ai tools for smolagents framework"""
import os
import tempfile
from io import BytesIO
from typing import Optional

import fal_client
import requests
from PIL import Image
from smolagents import Tool


class FalImageGenerationTool(Tool):
    """Tool for generating images using fal.ai models"""
    
    name = "fal_image_generation"
    description = """
    Generates images using fal.ai models. Supports multiple models and parameters.
    Use this tool when the user wants to create or generate new images from text descriptions.
    """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The text description of the image to generate"
        },
        "model": {
            "type": "string",
            "description": "Model to use: 'nano-banana', 'flux-schnell', or 'flux-pro'. Default: 'nano-banana'",
            "nullable": True
        },
        "width": {
            "type": "integer",
            "description": "Image width in pixels. Default: 720",
            "nullable": True
        },
        "height": {
            "type": "integer",
            "description": "Image height in pixels. Default: 720",
            "nullable": True
        },
        "num_images": {
            "type": "integer",
            "description": "Number of images to generate (1-4). Default: 1",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.models = {
            "nano-banana": "fal-ai/nano-banana",
            "flux-schnell": "fal-ai/flux/schnell",
            "flux-pro": "fal-ai/flux-pro"
        }

    def forward(
        self,
        prompt: str,
        model: str = "nano-banana",
        width: int = 720,
        height: int = 720,
        num_images: int = 1
    ) -> str:
        """Generate images using fal.ai
        
        Args:
            prompt: Text description of the image
            model: Model identifier
            width: Image width in pixels
            height: Image height in pixels
            num_images: Number of images to generate
            
        Returns:
            String describing the generated images and their paths
        """
        try:
            model_id = self.models.get(model, self.models["nano-banana"])

            args = {
                "prompt": prompt,
                "image_size": {"width": width, "height": height},
                "num_images": min(num_images, 4),
                "num_inference_steps": 12 if model == "nano-banana" else 28
            }

            handler = fal_client.submit(model_id, arguments=args)
            result = handler.get()

            # Download and save images
            image_paths = []
            if "images" in result:
                for idx, img_data in enumerate(result["images"]):
                    img_url = img_data.get("url")
                    if img_url:
                        response = requests.get(img_url)
                        img = Image.open(BytesIO(response.content))

                        # Save to temp file
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False, suffix=".png"
                        )
                        img.save(temp_file.name)
                        image_paths.append(temp_file.name)

            return f"Generated {len(image_paths)} image(s): {', '.join(image_paths)}"

        except Exception as e:
            return f"Error generating images: {str(e)}"


class FalVideoGenerationTool(Tool):
    """Tool for generating videos using fal.ai models"""
    
    name = "fal_video_generation"
    description = """
    Generates videos using fal.ai video models.
    Use this tool when the user wants to create videos or animations from text descriptions.
    """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The text description of the video to generate"
        },
        "duration": {
            "type": "integer",
            "description": "Video duration in seconds (3-10). Default: 5",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, prompt: str, duration: int = 5) -> str:
        """Generate video using fal.ai
        
        Args:
            prompt: Text description of the video
            duration: Video duration in seconds (3-10)
            
        Returns:
            String with video URL or error message
        """
        try:
            args = {
                "prompt": prompt,
                "duration": min(max(duration, 3), 10)
            }

            handler = fal_client.submit("fal-ai/luma-dream-machine", arguments=args)
            result = handler.get()

            if "video" in result:
                video_url = result["video"].get("url")
                return f"Generated video: {video_url}"

            return f"Video generation completed: {str(result)}"

        except Exception as e:
            return f"Error generating video: {str(e)}"


class FalImageEditTool(Tool):
    """Tool for editing images using fal.ai models"""
    
    name = "fal_image_edit"
    description = """
    Edits or modifies existing images using fal.ai models.
    Use this tool when the user wants to edit, modify, or transform an uploaded image.
    """
    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the image file to edit"
        },
        "prompt": {
            "type": "string",
            "description": "Description of the edits or transformations to apply"
        },
        "strength": {
            "type": "number",
            "description": "Edit strength (0.0-1.0). Default: 0.8",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, image_path: str, prompt: str, strength: float = 0.8) -> str:
        """Edit image using fal.ai
        
        Args:
            image_path: Path to the image file
            prompt: Description of edits to apply
            strength: Edit strength (0.0-1.0)
            
        Returns:
            String with edited image path or error message
        """
        try:
            # Upload image to fal.ai
            with open(image_path, 'rb') as f:
                image_url = fal_client.upload(f, "image/png")

            args = {
                "image_url": image_url,
                "prompt": prompt,
                "strength": strength
            }

            handler = fal_client.submit("fal-ai/flux/dev/image-to-image", arguments=args)
            result = handler.get()

            if "images" in result and result["images"]:
                edited_url = result["images"][0].get("url")

                # Download edited image
                response = requests.get(edited_url)
                img = Image.open(BytesIO(response.content))

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(temp_file.name)

                return f"Edited image saved to: {temp_file.name}"

            return f"Image editing completed: {str(result)}"

        except Exception as e:
            return f"Error editing image: {str(e)}"
