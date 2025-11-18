"""Custom fal.ai tools for smolagents framework"""
import os
import tempfile
import json
from io import BytesIO
from time import time
from typing import Optional

import fal_client
import requests
from PIL import Image
from smolagents import Tool


class FalImageGenerationTool(Tool):
    """Tool for generating images using fal.ai models"""

    name = "fal_image_generation"
    description = """
    Generates images using fal.ai models. Model is selected based on performance mode and the type of image being generated.
    Use this tool when the user wants to create or generate new images from text descriptions. Select based on user intent and pro vs fast model
    based on that intent.
    
    Anime: pro=seedream4, fast=seedream4
    Cartoon and Illustrations: pro=imagen4_ultra, fast=seedream4
    General and Photorealistic: pro=seedream4, fast=seedream4
    Graphic Design and Digital Rendering: pro=seedream4, fast=seedream4
    Traditional Art: pro=seedream4, fast=seedream4
    Vintage and Retro: pro=seedream4, fast=seedream4
    Commercial(ads, product shots): pro=imagen4_ultra, fast=imagen4_ultra
    Fantasy and Mythical: pro=seedream4, fast=seedream4
    Futuristic and Sci-Fi: pro=seedream4, fast=seedream4
    Nature and Landscapes: pro=seedream4, fast=seedream4
    People- Groups and Activities: pro=seedream4, fast=seedream4
    People- Portraits: pro=seedream4, fast=seedream4
    Physical Spaces: pro=seedream4, fast=seedream4
    Text and Typography: pro=imagen4_ultra, fast=nano-banana
    UI/UX: pro=imagen4_ultra, fast=nano-banana

    IMPORTANT: After calling this tool and receiving the generated image path, you MUST call final_answer()
    with the image path to complete the task. Do NOT call this tool multiple times for the same request.
    """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The text description of the image to generate"
        },
        "model": {
            "type": "string",
            "description": "model identifier (nano-banana, seedream4, imagen4_ultra). Default: seedream4",
            "nullable": False
        },
        "mode":{
            "type": "string",
            "description": "Performance mode (pro or fast). Default: fast",
            "nullable": False
        },
        "image_type":{
            "type": "string",
            "description": "Type of image to generate (e.g., anime, photorealistic, cartoon, graphic design, etc.). Default: general",
            "nullable": False
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
            "seedream4": "fal-ai/bytedance/seedream/v4/text-to-image",
            "imagen4_ultra": "fal-ai/imagen4/preview/ultra",
        }

    def forward(
        self,
        prompt: str,
        model: str = "seedream4",
        width: int = 720,
        height: int = 720,
        num_images: int = 1,
        mode: str = "fast",
        image_type: str = "general"
    ) -> str:
        """Generate images using fal.ai

        Args:
            prompt: Text description of the image
            model: Model identifier (ignored - uses config mode instead)
            width: Image width in pixels
            height: Image height in pixels
            num_images: Number of images to generate

        Returns:
            String describing the generated images and their paths
        """
        try:
            # Get model from config instead of parameter
            model_id = self.models[model]

            args = {
                "prompt": prompt,
                "image_size": {"width": width, "height": height},
                "num_images": min(num_images, 4),
            }

            # Print debug info
            debug_info = {
                "tool": "FalImageGenerationTool",
                "model": model_id,
                "model_id": model_id,
                "arguments": args
            }
            print(f"\n{'='*80}")
            print(f"DEBUG - Image Generation Tool Called:")
            print(json.dumps(debug_info, indent=2))
            print(f"{'='*80}\n")

            st = time()
            handler = fal_client.submit(model_id, arguments=args)
            result = handler.get()
            et = time()
            print(f"DEBUG - Image generation result: {result}")
            print(f"DEBUG - Time taken for image generation: {et - st} seconds")

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

            # # Return mock response for testing (uncomment API calls above when ready)
            # return f"[MOCK] Would generate {num_images} image(s) using model '{model_id}' in {mode} mode with type {image_type}'"

        except Exception as e:
            return f"Error generating images: {str(e)}"


class FalVideoGenerationTool(Tool):
    """Tool for generating videos using fal.ai models"""
    
    name = "fal_video_generation"
    description = """
    Generates videos using fal.ai video models.
    Use this tool when the user wants to create videos or animations from text descriptions.
    
    pro mode: veo3
    fast mode: veo3_fast

    IMPORTANT: After calling this tool and receiving the generated video path, you MUST call final_answer()
    with the video path to complete the task. Do NOT call this tool multiple times for the same request.
    """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The text description of the video to generate"
        },
        "duration": {
            "type": "integer",
            "description": "Video duration in seconds (4-8). Default: 6",
            "nullable": True
        },
        "model": {
            "type": "string",
            "description": "model identifier (veo3, veo3_fast). Default: veo3_fast",
            "nullable": False
        },
        "mode":{
            "type": "string",
            "description": "Performance mode (pro or fast). Default: fast",
            "nullable": False
        },
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.models = {
            "veo3": "fal-ai/veo3",
            "veo3_fast": "fal-ai/veo3/fast",
        }
        
    def forward(self, prompt: str, model:str="veo3_fast", mode:str="fast", duration: int = 6) -> str:
        """Generate video using fal.ai

        Args:
            prompt: Text description of the video
            duration: Video duration in seconds (3-10)

        Returns:
            String with local video file path or error message
        """
        try:
            # Get model from config
            # model_id = self.models[model]
            model_id = "fal-ai/bytedance/seedance/v1/lite/text-to-video"

            args = {
                "prompt": prompt,
                "duration": min(max(duration, 6), 6),
                "resolution": "720p"
            }

            # Print debug info
            debug_info = {
                "tool": "FalVideoGenerationTool",
                "mode": mode,
                "model_id": model_id,
                "arguments": args
            }
            print(f"\n{'='*80}")
            print(f"DEBUG - Video Generation Tool Called:")
            print(json.dumps(debug_info, indent=2))
            print(f"{'='*80}\n")

            st = time()
            handler = fal_client.submit(model_id, arguments=args)
            result = handler.get()
            et = time()
            print(f"DEBUG - Video generation result: {result}")
            print(f"DEBUG - Time taken for video generation: {et - st} seconds")

            if "video" in result:
                video_url = result["video"].get("url")

                # Download video to temp file
                response = requests.get(video_url)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_file.write(response.content)
                temp_file.close()

                return f"Generated video saved to: {temp_file.name}"

            # # Return mock response for testing (uncomment API calls above when ready)
            # return f"[MOCK] Would generate {duration}s video using model '{model_id}' in {mode} mode"

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
            # Get model from config
            model_id = "fal-ai/nano-banana/edit" #13.2s
            # model_id = "fal-ai/bytedance/seedream/v4/edit" #13.16s

            # Upload image to fal.ai
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_url = fal_client.upload(image_data, "image/png")

            args = {
                "image_urls": [image_url],
                "prompt": prompt,
                # "strength": strength,
                "width": 1024,
                "height": 1024
            }

            # Print debug info
            debug_info = {
                "tool": "FalImageEditTool",
                "model_id": model_id,
                "arguments": {
                    "prompt": prompt,
                    "image_url": image_url[:50] + "..."  # Truncate for readability
                }
            }
            print(f"\n{'='*80}")
            print(f"DEBUG - Image Edit Tool Called:")
            print(json.dumps(debug_info, indent=2))
            print(f"{'='*80}\n")

            st= time()
            handler = fal_client.submit(model_id, arguments=args)
            result = handler.get()
            et = time()
            print(f"DEBUG - Time taken for image editing: {et - st} seconds")
            print(f"DEBUG - Image editing result: {result}")

            if "images" in result and result["images"]:
                edited_url = result["images"][0].get("url")

                # Download edited image
                response = requests.get(edited_url)
                img = Image.open(BytesIO(response.content))

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(temp_file.name)

                return f"Edited image saved to: {temp_file.name}"

            # Return mock response for testing (uncomment API calls above when ready)
            # return f"[MOCK] Would edit image with prompt '{prompt}' using model '{model_id}'"

        except Exception as e:
            return f"Error editing image: {str(e)}"
