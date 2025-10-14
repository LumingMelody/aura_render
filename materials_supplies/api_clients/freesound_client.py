"""
Freesound API客户端 - 音效和音频素材
"""
from typing import Dict, List, Any, Optional
import asyncio
from .base_client import (
    BaseMaterialClient,
    MaterialSearchRequest,
    MaterialMetadata,
    SearchResponse
)


class FreesoundClient(BaseMaterialClient):
    """Freesound API客户端"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://freesound.org/apiv2",
            timeout=30
        )
        self.rate_limit_delay = 1.0  # 1 request per second for free tier

    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    async def search_materials(self, request: MaterialSearchRequest) -> SearchResponse:
        """搜索Freesound音频素材"""
        params = {
            "query": request.query,
            "page_size": min(request.limit, 150),  # Freesound max 150
            "page": (request.offset // request.limit) + 1,
            "fields": "id,name,description,url,preview-hq-mp3,preview-lq-mp3,images,duration,download,filesize,type,channels,bitrate,bitdepth,samplerate,username,license,tags,num_downloads,avg_rating,created"
        }

        # Add filters
        filters = []

        if request.duration_range:
            min_dur, max_dur = request.duration_range
            filters.append(f"duration:[{min_dur} TO {max_dur}]")

        if request.tags:
            # Use tags as additional filters
            for tag in request.tags:
                filters.append(f"tag:{tag}")

        # File type filter
        if request.format:
            if request.format.lower() in ["mp3", "wav", "ogg", "flac"]:
                filters.append(f"type:{request.format.lower()}")

        # Quality filters
        filters.append("channels:[1 TO 2]")  # Mono or stereo
        filters.append("samplerate:[22050 TO *]")  # Good quality

        if filters:
            params["filter"] = " ".join(filters)

        start_time = asyncio.get_event_loop().time()
        response = await self._make_request("GET", "/search/text/", params=params)
        search_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

        materials = []
        for sound in response.get("results", []):
            # Choose best preview URL
            preview_url = (sound.get("previews", {}).get("preview-hq-mp3") or
                          sound.get("previews", {}).get("preview-lq-mp3") or
                          "")

            # Get thumbnail
            thumbnail_url = ""
            images = sound.get("images", {})
            if images:
                thumbnail_url = (images.get("waveform_m") or
                               images.get("waveform_s") or
                               images.get("spectral_m") or
                               "")

            material = MaterialMetadata(
                id=str(sound["id"]),
                title=sound.get("name", f"Freesound {sound['id']}"),
                description=sound.get("description", "")[:500],  # Limit description length
                url=sound.get("download", ""),  # Download URL (requires authentication)
                thumbnail_url=thumbnail_url,
                duration=float(sound.get("duration", 0)),
                format=sound.get("type", "").lower(),
                size_bytes=sound.get("filesize", 0),
                tags=sound.get("tags", []) if isinstance(sound.get("tags"), list) else [],
                license_type=sound.get("license", ""),
                author=sound.get("username", "Freesound"),
                created_at=sound.get("created", ""),
                download_count=sound.get("num_downloads", 0),
                quality_score=self._calculate_quality_score(sound)
            )

            # Use preview URL for direct access
            if preview_url:
                material.url = preview_url

            materials.append(material)

        return SearchResponse(
            materials=materials,
            total_count=response.get("count", 0),
            page_size=len(materials),
            current_page=params["page"],
            has_more=response.get("next") is not None,
            search_time_ms=search_time
        )

    async def get_material_details(self, material_id: str) -> MaterialMetadata:
        """获取音频素材详情"""
        params = {
            "fields": "id,name,description,url,preview-hq-mp3,preview-lq-mp3,images,duration,download,filesize,type,channels,bitrate,bitdepth,samplerate,username,license,tags,num_downloads,avg_rating,created,similar_sounds"
        }

        response = await self._make_request("GET", f"/sounds/{material_id}/", params=params)

        # Parse response similar to search results
        sound = response
        preview_url = (sound.get("previews", {}).get("preview-hq-mp3") or
                      sound.get("previews", {}).get("preview-lq-mp3") or
                      "")

        thumbnail_url = ""
        images = sound.get("images", {})
        if images:
            thumbnail_url = (images.get("waveform_m") or
                           images.get("spectral_m") or
                           "")

        material = MaterialMetadata(
            id=str(sound["id"]),
            title=sound.get("name", f"Freesound {sound['id']}"),
            description=sound.get("description", ""),
            url=preview_url or sound.get("download", ""),
            thumbnail_url=thumbnail_url,
            duration=float(sound.get("duration", 0)),
            format=sound.get("type", "").lower(),
            size_bytes=sound.get("filesize", 0),
            tags=sound.get("tags", []) if isinstance(sound.get("tags"), list) else [],
            license_type=sound.get("license", ""),
            author=sound.get("username", "Freesound"),
            created_at=sound.get("created", ""),
            download_count=sound.get("num_downloads", 0),
            quality_score=self._calculate_quality_score(sound)
        )

        return material

    def _calculate_quality_score(self, sound: Dict) -> float:
        """计算音频质量分数"""
        score = 0.4  # Base score

        # Sample rate bonus
        samplerate = sound.get("samplerate", 0)
        if samplerate >= 44100:
            score += 0.2
        elif samplerate >= 22050:
            score += 0.1

        # Bit depth bonus
        bitdepth = sound.get("bitdepth", 0)
        if bitdepth >= 16:
            score += 0.1

        # Rating bonus
        rating = sound.get("avg_rating", 0)
        if rating >= 4:
            score += 0.2
        elif rating >= 3:
            score += 0.1

        # Download count bonus (popularity)
        downloads = sound.get("num_downloads", 0)
        if downloads > 1000:
            score += 0.1
        elif downloads > 100:
            score += 0.05

        return min(score, 1.0)

    async def search_by_category(self, category: str, limit: int = 20) -> List[MaterialMetadata]:
        """按类别搜索音频"""
        # Common audio categories for video production
        category_queries = {
            "music": "music loop background",
            "sfx": "sound effect",
            "nature": "nature ambient outdoor",
            "urban": "city urban street",
            "voice": "voice speech human",
            "mechanical": "machine mechanical industrial",
            "electronic": "electronic digital synthetic",
            "percussion": "drum percussion beat"
        }

        query = category_queries.get(category.lower(), category)

        request = MaterialSearchRequest(
            query=query,
            content_type="audio",
            limit=limit
        )

        response = await self.search_materials(request)
        return response.materials

    async def get_similar_sounds(self, sound_id: str, limit: int = 10) -> List[MaterialMetadata]:
        """获取相似音频"""
        try:
            params = {
                "fields": "id,name,description,preview-hq-mp3,preview-lq-mp3,duration,username,tags",
                "page_size": min(limit, 150)
            }

            response = await self._make_request("GET", f"/sounds/{sound_id}/similar/", params=params)

            materials = []
            for sound in response.get("results", []):
                preview_url = (sound.get("previews", {}).get("preview-hq-mp3") or
                              sound.get("previews", {}).get("preview-lq-mp3") or
                              "")

                material = MaterialMetadata(
                    id=str(sound["id"]),
                    title=sound.get("name", f"Similar Sound {sound['id']}"),
                    description=sound.get("description", "")[:200],
                    url=preview_url,
                    thumbnail_url="",
                    duration=float(sound.get("duration", 0)),
                    tags=sound.get("tags", []) if isinstance(sound.get("tags"), list) else [],
                    author=sound.get("username", "Freesound"),
                    quality_score=0.7  # Default for similar sounds
                )
                materials.append(material)

            return materials

        except Exception as e:
            print(f"Error getting similar sounds: {e}")
            return []

    async def search_by_duration(self, min_duration: float, max_duration: float, query: str = "", limit: int = 20) -> List[MaterialMetadata]:
        """按时长搜索音频"""
        request = MaterialSearchRequest(
            query=query,
            duration_range=(min_duration, max_duration),
            content_type="audio",
            limit=limit
        )

        response = await self.search_materials(request)
        return response.materials

    async def get_user_sounds(self, username: str, limit: int = 20) -> List[MaterialMetadata]:
        """获取指定用户的音频"""
        try:
            params = {
                "fields": "id,name,description,preview-hq-mp3,preview-lq-mp3,duration,tags,num_downloads",
                "page_size": min(limit, 150)
            }

            response = await self._make_request("GET", f"/users/{username}/sounds/", params=params)

            materials = []
            for sound in response.get("results", []):
                preview_url = (sound.get("previews", {}).get("preview-hq-mp3") or
                              sound.get("previews", {}).get("preview-lq-mp3") or
                              "")

                material = MaterialMetadata(
                    id=str(sound["id"]),
                    title=sound.get("name", f"Sound {sound['id']}"),
                    description=sound.get("description", "")[:200],
                    url=preview_url,
                    thumbnail_url="",
                    duration=float(sound.get("duration", 0)),
                    tags=sound.get("tags", []) if isinstance(sound.get("tags"), list) else [],
                    author=username,
                    download_count=sound.get("num_downloads", 0),
                    quality_score=0.7
                )
                materials.append(material)

            return materials

        except Exception as e:
            print(f"Error getting user sounds: {e}")
            return []