"""
Audio Material Providers

Implementations for various audio/music material providers.
"""

import httpx
from typing import List, Dict, Any, Optional

from .base import (
    MaterialProvider, MaterialSearchQuery, MaterialSearchResponse, 
    MaterialSearchResult, MaterialMetadata, MaterialType, MaterialFormat,
    MaterialLicense
)


class FreeSoundProvider(MaterialProvider):
    """
    Freesound.org API provider for audio effects and samples
    
    Requires FREESOUND_API_KEY environment variable or in config
    Free API: 2000 requests per day
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("freesound", config)
        self.api_key = self.config.get("api_key") or self.config.get("FREESOUND_API_KEY")
        self.base_url = "https://freesound.org/apiv2"
        
        if not self.api_key:
            self.logger.warning("Freesound API key not configured")
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_supported_types(self) -> List[MaterialType]:
        return [MaterialType.AUDIO]
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        return {
            "requests_per_day": 2000,
            "concurrent_requests": 3
        }
    
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Search Freesound for audio"""
        import time
        start_time = time.time()
        
        if not self.is_available():
            return MaterialSearchResponse(
                query=query,
                results=[],
                total_count=0,
                search_time=0.0,
                provider="freesound",
                provider_info={"error": "API key not configured"}
            )
        
        search_term = " ".join(query.keywords)
        params = {
            "query": search_term,
            "page_size": min(query.limit, 150),
            "page": (query.offset // query.limit) + 1 if query.limit > 0 else 1,
            "fields": "id,name,description,tags,duration,filesize,type,channels,bitrate,samplerate,username,license,previews",
            "token": self.api_key
        }
        
        # Add duration filter
        filters = []
        if query.min_duration:
            filters.append(f"duration:[{query.min_duration} TO *]")
        if query.max_duration:
            filters.append(f"duration:[* TO {query.max_duration}]")
        
        if filters:
            params["filter"] = " ".join(filters)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/search/text/",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                for sound_data in data.get("results", []):
                    result = self._convert_freesound_audio(sound_data, query.keywords)
                    if result:
                        results.append(result)
                
                search_time = time.time() - start_time
                
                return MaterialSearchResponse(
                    query=query,
                    results=results,
                    total_count=data.get("count", len(results)),
                    search_time=search_time,
                    provider="freesound",
                    provider_info={"next": data.get("next"), "previous": data.get("previous")}
                )
        
        except Exception as e:
            self.logger.error(f"Freesound search failed: {e}")
            return MaterialSearchResponse(
                query=query,
                results=[],
                total_count=0,
                search_time=time.time() - start_time,
                provider="freesound",
                provider_info={"error": str(e)}
            )
    
    def _convert_freesound_audio(self, sound_data: Dict[str, Any], search_keywords: List[str]) -> Optional[MaterialSearchResult]:
        """Convert Freesound data to MaterialSearchResult"""
        try:
            sound_id = str(sound_data.get("id"))
            
            # Get preview URL
            previews = sound_data.get("previews", {})
            preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
            
            # Parse tags
            tags = sound_data.get("tags", [])[:10]
            
            # Determine license
            license_name = sound_data.get("license", "")
            if "cc0" in license_name.lower():
                license_type = MaterialLicense.CREATIVE_COMMONS
            else:
                license_type = MaterialLicense.CREATIVE_COMMONS
            
            metadata = MaterialMetadata(
                title=sound_data.get("name", f"Audio {sound_id}"),
                description=sound_data.get("description", "")[:500],
                tags=tags,
                duration=float(sound_data.get("duration", 0)),
                format=MaterialFormat.MP3,
                file_size=sound_data.get("filesize"),
                bitrate=sound_data.get("bitrate"),
                license=license_type,
                license_url=f"https://freesound.org/people/{sound_data.get('username', '')}/sounds/{sound_id}/",
                source="Freesound",
                author=sound_data.get("username", "Unknown")
            )
            
            # Calculate relevance based on tags and search terms
            relevance_score = 0.5
            matching_tags = []
            for keyword in search_keywords:
                keyword_lower = keyword.lower()
                for tag in tags:
                    if keyword_lower in tag.lower():
                        relevance_score += 0.1
                        matching_tags.append(tag)
                        break
            
            relevance_score = min(1.0, relevance_score)
            
            return MaterialSearchResult(
                material_id=f"freesound_{sound_id}",
                material_type=MaterialType.AUDIO,
                url=f"https://freesound.org/people/{sound_data.get('username', '')}/sounds/{sound_id}/download/",
                preview_url=preview_url,
                metadata=metadata,
                relevance_score=relevance_score,
                matching_tags=matching_tags[:5],
                quality_score=min(1.0, (sound_data.get("bitrate", 128) / 320) * 0.7 + 0.3),
                popularity_score=0.6
            )
        
        except Exception as e:
            self.logger.error(f"Failed to convert Freesound data: {e}")
            return None
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Get detailed info about a Freesound audio"""
        if not material_id.startswith("freesound_"):
            return None
        
        sound_id = material_id.replace("freesound_", "")
        
        if not self.is_available():
            return None
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/sounds/{sound_id}/",
                    params={"token": self.api_key}
                )
                response.raise_for_status()
                sound_data = response.json()
                
                return self._convert_freesound_audio(sound_data, ["audio"])
        
        except Exception as e:
            self.logger.error(f"Failed to get Freesound audio info: {e}")
            return None
    
    async def download_material(self, material: MaterialSearchResult, local_path: str) -> bool:
        """Download audio from Freesound (requires special download URL)"""
        # Note: Freesound requires OAuth for downloads, this is simplified
        try:
            headers = {"Authorization": f"Token {self.api_key}"}
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(material.url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                
                with open(local_path, "wb") as f:
                    async for chunk in response.aiter_bytes(8192):
                        f.write(chunk)
                
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to download audio: {e}")
            return False
    
    async def get_download_url(self, material: MaterialSearchResult) -> str:
        """Get direct download URL"""
        return material.url


class MockAudioProvider(MaterialProvider):
    """Mock audio provider for development"""
    
    def __init__(self):
        super().__init__("mock_audio")
    
    async def search(self, query: MaterialSearchQuery) -> MaterialSearchResponse:
        """Return mock audio results"""
        import time
        start_time = time.time()
        
        results = []
        audio_types = ["music", "sfx", "voice", "ambient", "loop"]
        
        for i in range(min(query.limit, 3)):
            audio_type = audio_types[i % len(audio_types)]
            material_id = f"mock_audio_{audio_type}_{i}"
            
            metadata = MaterialMetadata(
                title=f"Mock {audio_type.title()}: {' '.join(query.keywords[:2])}",
                description=f"Mock {audio_type} content for {', '.join(query.keywords)}",
                tags=query.keywords + [audio_type],
                duration=30.0 if audio_type != "loop" else 10.0,
                format=MaterialFormat.MP3,
                bitrate=192,
                license=MaterialLicense.FREE,
                source="Mock Audio Provider"
            )
            
            result = MaterialSearchResult(
                material_id=material_id,
                material_type=MaterialType.AUDIO,
                url=f"https://mock-audio.com/{material_id}.mp3",
                preview_url=f"https://mock-audio.com/preview/{material_id}.mp3",
                metadata=metadata,
                relevance_score=max(0.4, 1.0 - i * 0.15),
                matching_tags=query.keywords[:3],
                quality_score=0.8,
                popularity_score=0.5
            )
            results.append(result)
        
        return MaterialSearchResponse(
            query=query,
            results=results,
            total_count=len(results),
            search_time=time.time() - start_time,
            provider="mock_audio",
            provider_info={"mock": True}
        )
    
    async def get_material_info(self, material_id: str) -> Optional[MaterialSearchResult]:
        """Return mock audio info"""
        if not material_id.startswith("mock_audio_"):
            return None
        
        metadata = MaterialMetadata(
            title=f"Mock Audio {material_id}",
            description="Mock audio for testing",
            tags=["mock", "test", "audio"],
            duration=30.0,
            format=MaterialFormat.MP3,
            license=MaterialLicense.FREE
        )
        
        return MaterialSearchResult(
            material_id=material_id,
            material_type=MaterialType.AUDIO,
            url=f"https://mock-audio.com/{material_id}.mp3",
            metadata=metadata,
            relevance_score=0.8,
            quality_score=0.8,
            popularity_score=0.5
        )
    
    async def download_material(self, material: MaterialSearchResult, local_path: str) -> bool:
        """Mock download"""
        try:
            with open(local_path, 'w') as f:
                f.write(f"Mock audio content for {material.material_id}")
            return True
        except Exception as e:
            self.logger.error(f"Mock audio download failed: {e}")
            return False
    
    async def get_download_url(self, material: MaterialSearchResult) -> str:
        """Return the mock URL"""
        return material.url


# Alias for the primary audio provider
AudioMaterialProvider = MockAudioProvider  # Switch to FreeSoundProvider when API key is available