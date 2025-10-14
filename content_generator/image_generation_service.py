"""
AI Image Generation Service

Provides AI-powered image generation capabilities using multiple providers:
- OpenAI DALL-E 3
- Stability AI Stable Diffusion
- Midjourney (via API when available)
- Local Stable Diffusion models

Supports various image generation scenarios:
- Scene generation from text descriptions
- Style transfer and variations
- Concept art and storyboarding
- Brand-consistent imagery
"""

import asyncio
import base64
import hashlib
import os
import tempfile
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging
import httpx
from PIL import Image
import io

from config import Settings
from monitoring import get_error_handler, get_metrics_collector
from monitoring.error_handler import ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

class ImageProvider(Enum):
    """Supported image generation providers"""
    OPENAI_DALLE = "openai_dalle"
    STABILITY_AI = "stability_ai"
    MIDJOURNEY = "midjourney"
    LOCAL_SD = "local_stable_diffusion"

class ImageStyle(Enum):
    """Image generation styles"""
    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic"
    CARTOON = "cartoon"
    CINEMATIC = "cinematic"
    MINIMALIST = "minimalist"
    VINTAGE = "vintage"
    FUTURISTIC = "futuristic"
    CORPORATE = "corporate"

class ImageSize(Enum):
    """Standard image sizes"""
    SQUARE_1024 = "1024x1024"
    LANDSCAPE_1792_1024 = "1792x1024"
    PORTRAIT_1024_1792 = "1024x1792"
    HD_1920_1080 = "1920x1080"
    VERTICAL_1080_1920 = "1080x1920"

@dataclass
class ImageGenerationRequest:
    """Image generation request parameters"""
    prompt: str
    style: ImageStyle = ImageStyle.PHOTOREALISTIC
    size: ImageSize = ImageSize.SQUARE_1024
    quality: str = "standard"  # standard, hd
    provider: Optional[ImageProvider] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    cfg_scale: float = 7.0
    steps: int = 20
    batch_size: int = 1

@dataclass
class GeneratedImage:
    """Generated image result"""
    image_path: str
    thumbnail_path: Optional[str]
    prompt: str
    provider: ImageProvider
    style: ImageStyle
    size: str
    generation_time: float
    cost: float
    metadata: Dict[str, Any]

class BaseImageProvider:
    """Base class for image generation providers"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    async def generate_image(
        self,
        request: ImageGenerationRequest,
        output_path: str
    ) -> Optional[GeneratedImage]:
        """Generate image from text prompt"""
        raise NotImplementedError
        
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        return True
        
    def get_supported_sizes(self) -> List[ImageSize]:
        """Get supported image sizes for this provider"""
        return list(ImageSize)

class OpenAIDALLEProvider(BaseImageProvider):
    """OpenAI DALL-E image generation provider"""
    
    def __init__(self, api_key: str):
        super().__init__("openai_dalle")
        self.api_key = api_key
        self.client = None
        
        if api_key:
            try:
                import openai
                self.client = openai.AsyncOpenAI(api_key=api_key)
                self.logger.info("OpenAI DALL-E provider initialized")
            except ImportError:
                self.logger.error("OpenAI library not installed")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
    
    async def generate_image(
        self,
        request: ImageGenerationRequest,
        output_path: str
    ) -> Optional[GeneratedImage]:
        """Generate image using DALL-E 3"""
        
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        start_time = time.time()
        
        try:
            # Prepare DALL-E request
            dalle_request = {
                "model": "dall-e-3",
                "prompt": self._enhance_prompt(request.prompt, request.style),
                "size": self._convert_size(request.size),
                "quality": request.quality,
                "n": 1,  # DALL-E 3 only supports n=1
                "response_format": "url"
            }
            
            self.logger.info(f"Generating image with DALL-E: {request.prompt[:50]}...")
            
            # Call DALL-E API
            response = await self.client.images.generate(**dalle_request)
            
            if not response.data:
                raise RuntimeError("No images generated")
            
            image_data = response.data[0]
            
            # Download and save image
            async with httpx.AsyncClient() as client:
                image_response = await client.get(image_data.url)
                image_response.raise_for_status()
                
                # Save image
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(image_response.content)
            
            # Create thumbnail
            thumbnail_path = self._create_thumbnail(output_path)
            
            generation_time = time.time() - start_time
            
            # Calculate estimated cost (DALL-E 3 pricing)
            cost = self._calculate_dalle_cost(request.quality, request.size)
            
            result = GeneratedImage(
                image_path=output_path,
                thumbnail_path=thumbnail_path,
                prompt=request.prompt,
                provider=ImageProvider.OPENAI_DALLE,
                style=request.style,
                size=request.size.value,
                generation_time=generation_time,
                cost=cost,
                metadata={
                    "model": "dall-e-3",
                    "revised_prompt": image_data.revised_prompt,
                    "quality": request.quality,
                    "original_size": dalle_request["size"]
                }
            )
            
            self.logger.info(f"DALL-E image generated successfully in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"DALL-E generation failed: {str(e)}")
            raise e
    
    def _enhance_prompt(self, prompt: str, style: ImageStyle) -> str:
        """Enhance prompt with style-specific descriptors"""
        
        style_enhancers = {
            ImageStyle.PHOTOREALISTIC: "photorealistic, high resolution, professional photography",
            ImageStyle.ARTISTIC: "artistic painting, creative interpretation, expressive",
            ImageStyle.CARTOON: "cartoon style, animated, colorful illustration",
            ImageStyle.CINEMATIC: "cinematic lighting, movie still, dramatic composition",
            ImageStyle.MINIMALIST: "minimalist design, clean, simple, modern",
            ImageStyle.VINTAGE: "vintage style, retro aesthetic, aged appearance",
            ImageStyle.FUTURISTIC: "futuristic design, sci-fi aesthetic, modern technology",
            ImageStyle.CORPORATE: "professional, business setting, clean corporate style"
        }
        
        enhancer = style_enhancers.get(style, "")
        if enhancer:
            return f"{prompt}, {enhancer}"
        return prompt
    
    def _convert_size(self, size: ImageSize) -> str:
        """Convert internal size enum to DALL-E format"""
        size_mapping = {
            ImageSize.SQUARE_1024: "1024x1024",
            ImageSize.LANDSCAPE_1792_1024: "1792x1024",
            ImageSize.PORTRAIT_1024_1792: "1024x1792",
            ImageSize.HD_1920_1080: "1024x1024",  # DALL-E doesn't support 1920x1080
            ImageSize.VERTICAL_1080_1920: "1024x1792"
        }
        return size_mapping.get(size, "1024x1024")
    
    def _calculate_dalle_cost(self, quality: str, size: ImageSize) -> float:
        """Calculate estimated DALL-E generation cost"""
        # DALL-E 3 pricing (as of 2024)
        if quality == "hd":
            if size in [ImageSize.LANDSCAPE_1792_1024, ImageSize.PORTRAIT_1024_1792]:
                return 0.080  # $0.080 per image for HD 1792x1024 or 1024x1792
            else:
                return 0.080  # $0.080 per image for HD 1024x1024
        else:
            if size in [ImageSize.LANDSCAPE_1792_1024, ImageSize.PORTRAIT_1024_1792]:
                return 0.040  # $0.040 per image for standard 1792x1024 or 1024x1792
            else:
                return 0.040  # $0.040 per image for standard 1024x1024
    
    def _create_thumbnail(self, image_path: str, size: Tuple[int, int] = (256, 256)) -> str:
        """Create thumbnail of generated image"""
        try:
            thumbnail_path = image_path.replace('.png', '_thumb.jpg').replace('.jpg', '_thumb.jpg')
            
            with Image.open(image_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.convert('RGB').save(thumbnail_path, 'JPEG', quality=85)
            
            return thumbnail_path
        except Exception as e:
            self.logger.error(f"Failed to create thumbnail: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if OpenAI DALL-E is available"""
        return self.client is not None
    
    def get_supported_sizes(self) -> List[ImageSize]:
        """Get DALL-E supported sizes"""
        return [
            ImageSize.SQUARE_1024,
            ImageSize.LANDSCAPE_1792_1024,
            ImageSize.PORTRAIT_1024_1792
        ]

class StabilityAIProvider(BaseImageProvider):
    """Stability AI Stable Diffusion provider"""
    
    def __init__(self, api_key: str):
        super().__init__("stability_ai")
        self.api_key = api_key
        self.base_url = "https://api.stability.ai"
        
    async def generate_image(
        self,
        request: ImageGenerationRequest,
        output_path: str
    ) -> Optional[GeneratedImage]:
        """Generate image using Stability AI"""
        
        if not self.api_key:
            raise RuntimeError("Stability AI API key not configured")
        
        start_time = time.time()
        
        try:
            # Prepare Stability AI request
            width, height = self._parse_size(request.size)
            
            payload = {
                "text_prompts": [
                    {
                        "text": self._enhance_prompt(request.prompt, request.style),
                        "weight": 1.0
                    }
                ],
                "cfg_scale": request.cfg_scale,
                "height": height,
                "width": width,
                "steps": request.steps,
                "samples": request.batch_size,
                "style_preset": self._get_style_preset(request.style)
            }
            
            if request.negative_prompt:
                payload["text_prompts"].append({
                    "text": request.negative_prompt,
                    "weight": -1.0
                })
            
            if request.seed:
                payload["seed"] = request.seed
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            self.logger.info(f"Generating image with Stability AI: {request.prompt[:50]}...")
            
            # Call Stability AI API
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                result_data = response.json()
            
            if not result_data.get("artifacts"):
                raise RuntimeError("No images generated")
            
            # Save first image
            artifact = result_data["artifacts"][0]
            image_data = base64.b64decode(artifact["base64"])
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            # Create thumbnail
            thumbnail_path = self._create_thumbnail(output_path)
            
            generation_time = time.time() - start_time
            
            # Calculate estimated cost
            cost = self._calculate_stability_cost(width, height, request.steps)
            
            result = GeneratedImage(
                image_path=output_path,
                thumbnail_path=thumbnail_path,
                prompt=request.prompt,
                provider=ImageProvider.STABILITY_AI,
                style=request.style,
                size=request.size.value,
                generation_time=generation_time,
                cost=cost,
                metadata={
                    "model": "stable-diffusion-xl-1024-v1-0",
                    "cfg_scale": request.cfg_scale,
                    "steps": request.steps,
                    "seed": artifact.get("seed"),
                    "finish_reason": artifact.get("finishReason")
                }
            )
            
            self.logger.info(f"Stability AI image generated successfully in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Stability AI generation failed: {str(e)}")
            raise e
    
    def _parse_size(self, size: ImageSize) -> Tuple[int, int]:
        """Parse size enum to width, height"""
        size_mapping = {
            ImageSize.SQUARE_1024: (1024, 1024),
            ImageSize.LANDSCAPE_1792_1024: (1792, 1024),
            ImageSize.PORTRAIT_1024_1792: (1024, 1792),
            ImageSize.HD_1920_1080: (1920, 1080),
            ImageSize.VERTICAL_1080_1920: (1080, 1920)
        }
        return size_mapping.get(size, (1024, 1024))
    
    def _get_style_preset(self, style: ImageStyle) -> Optional[str]:
        """Map internal style to Stability AI style preset"""
        style_mapping = {
            ImageStyle.PHOTOREALISTIC: "photographic",
            ImageStyle.ARTISTIC: "artistic",
            ImageStyle.CARTOON: "comic-book",
            ImageStyle.CINEMATIC: "cinematic",
            ImageStyle.MINIMALIST: "minimalist",
            ImageStyle.VINTAGE: "analog-film",
            ImageStyle.FUTURISTIC: "digital-art"
        }
        return style_mapping.get(style)
    
    def _enhance_prompt(self, prompt: str, style: ImageStyle) -> str:
        """Enhance prompt for Stability AI"""
        # Stability AI specific enhancements
        style_enhancers = {
            ImageStyle.PHOTOREALISTIC: "masterpiece, best quality, ultra detailed, 8k resolution",
            ImageStyle.ARTISTIC: "beautiful artwork, masterpiece, trending on artstation",
            ImageStyle.CARTOON: "cartoon style, vibrant colors, clean lines",
            ImageStyle.CINEMATIC: "cinematic shot, dramatic lighting, film photography",
            ImageStyle.MINIMALIST: "minimalist, clean, simple composition",
            ImageStyle.VINTAGE: "vintage photography, film grain, nostalgic",
            ImageStyle.FUTURISTIC: "futuristic, sci-fi, cyberpunk aesthetic",
            ImageStyle.CORPORATE: "professional, business, corporate photography"
        }
        
        enhancer = style_enhancers.get(style, "high quality, detailed")
        return f"{prompt}, {enhancer}"
    
    def _calculate_stability_cost(self, width: int, height: int, steps: int) -> float:
        """Calculate estimated Stability AI cost"""
        # Stability AI pricing is based on image resolution and steps
        pixel_count = width * height
        base_cost = 0.002  # Base cost per generation
        
        # Higher resolution multiplier
        if pixel_count > 1024 * 1024:
            base_cost *= (pixel_count / (1024 * 1024))
        
        # Steps multiplier (rough estimate)
        if steps > 20:
            base_cost *= (steps / 20)
        
        return round(base_cost, 4)
    
    def _create_thumbnail(self, image_path: str, size: Tuple[int, int] = (256, 256)) -> str:
        """Create thumbnail of generated image"""
        try:
            thumbnail_path = image_path.replace('.png', '_thumb.jpg')
            
            with Image.open(image_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.convert('RGB').save(thumbnail_path, 'JPEG', quality=85)
            
            return thumbnail_path
        except Exception as e:
            self.logger.error(f"Failed to create thumbnail: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Stability AI is available"""
        return bool(self.api_key)

class ImageGenerationService:
    """Main image generation service managing multiple providers"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)
        self.error_handler = get_error_handler()
        self.metrics = get_metrics_collector()
        
        self.providers: Dict[ImageProvider, BaseImageProvider] = {}
        self._initialize_providers()
        
    def _initialize_providers(self):
        """Initialize available image generation providers"""
        
        # OpenAI DALL-E
        if hasattr(self.settings, 'openai_api_key') and self.settings.openai_api_key:
            provider = OpenAIDALLEProvider(self.settings.openai_api_key)
            if provider.is_available():
                self.providers[ImageProvider.OPENAI_DALLE] = provider
                self.logger.info("OpenAI DALL-E provider initialized")
        
        # Stability AI
        if hasattr(self.settings, 'stability_api_key') and self.settings.stability_api_key:
            provider = StabilityAIProvider(self.settings.stability_api_key)
            if provider.is_available():
                self.providers[ImageProvider.STABILITY_AI] = provider
                self.logger.info("Stability AI provider initialized")
        
        if not self.providers:
            self.logger.warning("No image generation providers available")
    
    async def generate_image(
        self,
        prompt: str,
        style: ImageStyle = ImageStyle.PHOTOREALISTIC,
        size: ImageSize = ImageSize.SQUARE_1024,
        provider: Optional[ImageProvider] = None,
        quality: str = "standard",
        output_path: Optional[str] = None,
        **kwargs
    ) -> Optional[GeneratedImage]:
        """Generate image with specified parameters"""
        
        start_time = time.time()
        
        try:
            # Select provider
            if provider and provider in self.providers:
                selected_provider = self.providers[provider]
            else:
                # Use first available provider
                if not self.providers:
                    raise ValueError("No image generation providers available")
                selected_provider = list(self.providers.values())[0]
                provider = list(self.providers.keys())[0]
            
            # Generate output path if not provided
            if not output_path:
                output_dir = self.settings.temp_dir / "generated_images"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create filename from prompt hash
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                timestamp = int(time.time())
                output_path = str(output_dir / f"img_{timestamp}_{prompt_hash}_{style.value}.png")
            
            # Create generation request
            request = ImageGenerationRequest(
                prompt=prompt,
                style=style,
                size=size,
                quality=quality,
                provider=provider,
                **kwargs
            )
            
            self.logger.info(f"Generating image with {provider.value}: {prompt[:50]}...")
            
            # Generate image
            result = await selected_provider.generate_image(request, output_path)
            
            if result:
                duration = time.time() - start_time
                
                # Record metrics
                self.metrics.record_ai_service_call(
                    service="image_generation",
                    model=provider.value,
                    duration=duration,
                    success=True
                )
                
                self.logger.info(f"Image generated successfully in {duration:.2f}s: {output_path}")
                return result
            else:
                raise RuntimeError("Image generation returned no result")
        
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error
            await self.error_handler.handle_error(
                exception=e,
                category=ErrorCategory.AI_SERVICE,
                severity=ErrorSeverity.MEDIUM,
                context={
                    "service": "image_generation",
                    "prompt": prompt[:100],
                    "style": style.value,
                    "provider": provider.value if provider else "auto"
                }
            )
            
            # Record metrics
            self.metrics.record_ai_service_call(
                service="image_generation",
                model=provider.value if provider else "unknown",
                duration=duration,
                success=False
            )
            
            self.logger.error(f"Image generation failed: {str(e)}")
            return None
    
    async def generate_scene_images(
        self,
        scene_descriptions: List[str],
        style: ImageStyle = ImageStyle.CINEMATIC,
        size: ImageSize = ImageSize.LANDSCAPE_1792_1024,
        output_dir: Optional[str] = None
    ) -> List[Optional[GeneratedImage]]:
        """Generate multiple scene images for video storyboarding"""
        
        if not output_dir:
            output_dir = self.settings.temp_dir / "scene_images"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        tasks = []
        for i, description in enumerate(scene_descriptions):
            output_path = str(Path(output_dir) / f"scene_{i+1:03d}.png")
            
            task = self.generate_image(
                prompt=description,
                style=style,
                size=size,
                output_path=output_path
            )
            tasks.append(task)
        
        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(2)  # Max 2 concurrent generations
        
        async def limited_generate(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[limited_generate(task) for task in tasks],
            return_exceptions=True
        )
        
        # Filter out exceptions
        filtered_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Scene generation failed: {result}")
                filtered_results.append(None)
            else:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_available_providers(self) -> List[ImageProvider]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def get_provider_info(self, provider: ImageProvider) -> Dict[str, Any]:
        """Get information about a specific provider"""
        if provider not in self.providers:
            return {"available": False}
        
        provider_obj = self.providers[provider]
        return {
            "available": True,
            "name": provider.value,
            "supported_sizes": [size.value for size in provider_obj.get_supported_sizes()]
        }
    
    async def test_generation(self, provider: Optional[ImageProvider] = None) -> Dict[str, Any]:
        """Test image generation with sample prompt"""
        
        test_prompt = "A beautiful sunset over mountains, peaceful landscape"
        
        try:
            result = await self.generate_image(
                prompt=test_prompt,
                style=ImageStyle.PHOTOREALISTIC,
                size=ImageSize.SQUARE_1024,
                provider=provider
            )
            
            if result:
                return {
                    "success": True,
                    "image_path": result.image_path,
                    "generation_time": result.generation_time,
                    "cost": result.cost,
                    "provider": result.provider.value
                }
            else:
                return {
                    "success": False,
                    "error": "No result returned"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Global service instance
_image_service: Optional[ImageGenerationService] = None

def get_image_generation_service(settings: Optional[Settings] = None) -> ImageGenerationService:
    """Get global image generation service instance"""
    global _image_service
    if _image_service is None:
        _image_service = ImageGenerationService(settings)
    return _image_service