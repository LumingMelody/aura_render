# matcher/video_matcher.py
import httpx
from typing import List, Dict, Any, Optional
from ..models import VideoRequest, VideoResponse
try:
    from llm.qwen import QwenLLM
    qwen_client = QwenLLM()
except ImportError:
    qwen_client = None

try:
    from utils.color_analyzer import ColorStyleAnalyzer  # 假设已有该模块
except ImportError:
    ColorStyleAnalyzer = None

# -------------------------------
# 🎯 多级视频匹配主函数
# -------------------------------
import asyncio
import base64
from config import settings


class MainVideoMatcher:
    """主视频匹配器类"""

    def __init__(self):
        self.qwen_client = qwen_client
        self.color_analyzer = ColorStyleAnalyzer if ColorStyleAnalyzer else None

    async def match_videos(self, request: VideoRequest) -> VideoResponse:
        """匹配视频素材"""
        # 这是一个占位实现，实际逻辑需要根据需求完善
        return VideoResponse(
            status="success",
            materials=[],
            message="Video matching not yet implemented"
        )

# 全局 HTTP 客户端
async def get_http_client():
    return httpx.AsyncClient(timeout=30.0)

# -------------------------------
# 🚀 主函数：多级筛选 + AI 介入
# -------------------------------
async def match_main_video(request: VideoRequest) -> List[VideoResponse]:

    # Step 1: 获取候选
    candidates = await fetch_candidates_from_java(request)
    if not candidates:
        gen = await generate_video_by_ai(request, reason="no_candidates")
        return [gen] if gen else []

    # Step 2: 一级筛选 - 内容语义匹配（文本）
    content_matched = await filter_by_content_semantic(candidates, request.description)
    if not content_matched:
        gen = await generate_video_by_ai(request, reason="content_mismatch")
        return [gen] if gen else []

    # Step 3: 二级筛选 - 多模态图文一致性验证（Qwen-VL）
    visual_verified = await validate_with_qwen_vl(content_matched, request)
    if not visual_verified:
        # 图文内容不一致 → 无法修复 → 重新生成
        gen = await generate_video_by_ai(request, reason="visual_inconsistent")
        return [gen] if gen else []

    # Step 4: 三级决策 - 风格与色彩是否匹配？决定是直接返回、风格迁移、还是生成
    final_result = await decide_by_style_and_color(visual_verified, request)
    return [final_result] if final_result else []



    # Step 5: 色彩验证（可选）
    # final_candidates = await validate_color_style(visual_verified, request.category)
    # if not final_candidates:
    #     final_candidates = visual_verified  # 降级保留

    # # 转换为 VideoResponse
    # return [VideoResponse(
    #     url=c["url"],
    #     thumbnail=c["thumbnail"],
    #     in_point=0.0,
    #     out_point=min(c["duration"], request.duration),
    #     match_score=1.0
    # ) for c in final_candidates[:1]]


# -------------------------------
# 1️⃣ 从 Java 获取候选（真实API）
# -------------------------------
async def fetch_candidates_from_online(request: VideoRequest) -> List[Dict]:
    """调用真实素材库API获取视频候选"""
    
    # 导入provider manager
    from materials_supplies.providers.provider_manager import get_provider_manager
    from materials_supplies.providers.base_provider import MaterialType
    
    try:
        # 获取provider manager
        provider_manager = get_provider_manager()
        await provider_manager.initialize()
        
        # 构建搜索查询
        search_query = request.description
        if hasattr(request, 'keywords') and request.keywords:
            search_query = f"{search_query} {' '.join(request.keywords)}"
            
        # 搜索视频素材
        search_results = await provider_manager.search(
            query=search_query,
            material_type=MaterialType.VIDEO,
            limit=20,  # 获取更多候选以便筛选
            filters={
                "min_duration": getattr(request, 'duration', 10) * 0.5,  # 至少一半时长
                "max_duration": getattr(request, 'duration', 10) * 3,    # 最多3倍时长
            }
        )
        
        # 转换为内部格式
        candidates = []
        for result in search_results:
            candidates.append({
                "material_id": result.material_id,
                "url": result.url,
                "thumbnail": result.thumbnail_url or result.preview_url or "",
                "description": result.description or result.title,
                "tags": result.tags,
                "style": result.metadata.get("style", "通用"),
                "duration": result.duration or 30.0,
                "provider": result.provider,
                "relevance_score": result.relevance_score
            })
            
        # 如果没有找到真实素材，返回模拟数据作为fallback
        if not candidates and settings.development.enable_mock_services:
            return [
                {
                    "material_id": "mock_vid_001",
                    "url": "https://video.com/flying-car-tech.mp4",
                    "thumbnail": "https://thumb.com/flying-car.jpg",
                    "description": "一辆银色飞行汽车在霓虹科技城市上空飞行",
                    "tags": ["飞行汽车", "未来城市", "科技感"],
                    "style": "科技感",
                    "duration": 60.0,
                    "provider": "mock",
                    "relevance_score": 0.5
                }
            ]
            
        return candidates
        
    except Exception as e:
        # 记录错误并返回模拟数据
        print(f"Error fetching materials from providers: {e}")
        
        # Fallback to mock data in development
        if settings.development.enable_mock_services:
            return [
                {
                    "material_id": "fallback_vid_001",
                    "url": "https://video.com/default-tech.mp4",
                    "thumbnail": "https://thumb.com/default.jpg",
                    "description": "默认科技视频素材",
                    "tags": ["科技", "默认"],
                    "style": "通用",
                    "duration": 30.0,
                    "provider": "mock",
                    "relevance_score": 0.3
                }
            ]
        
        return []


# -------------------------------
# 2️⃣ 一级筛选：描述语义匹配
# -------------------------------
async def filter_by_content_semantic(candidates: List[Dict], user_desc: str) -> List[Dict]:
    """
    使用 Qwen 判断候选描述是否与用户需求内容一致
    """
    results = []
    async for c in candidates:
        prompt = f"""
        请判断以下两个描述是否表达相同或高度相似的内容：
        【用户需求】{user_desc}
        【素材描述】{c['description']}
        请输出 JSON：{{"match": true}} 或 {{"match": false}}
        """
        resp = await qwen_generate(prompt, parse_json=True)
        if resp and resp.get("match", False):
            results.append(c)
    return results


# -------------------------------
# 3️⃣ 二级筛选：风格标签匹配
# -------------------------------
def filter_by_style(candidates: List[Dict], required_style: str) -> List[Dict]:
    """
    简单风格关键词匹配
    """
    filtered = []
    req_lower = required_style.lower()
    for c in candidates:
        style = c.get("style", "").lower()
        tags = [t.lower() for t in c.get("tags", [])]
        if req_lower in style or any(req_lower in tag for tag in tags):
            filtered.append(c)
    return filtered


# -------------------------------
# 4️⃣ 三级筛选：色彩风格分析打分
# -------------------------------
async def score_with_color_analysis(candidates: List[Dict], request: VideoRequest) -> List[Dict]:
    """
    使用 ColorStyleAnalyzer 分析缩略图色彩，匹配风格
    示例：科技感 → 蓝/银；复古 → 棕/黄；赛博朋克 → 紫/粉
    """
    style_color_map = {
        "科技感": ["blue", "silver", "cyan"],
        "未来感": ["white", "blue", "purple"],
        "复古": ["brown", "yellow", "beige"],
        "赛博朋克": ["pink", "purple", "neon"]
    }

    target_colors = style_color_map.get(request.category.lower(), [])

    if not target_colors:
        return candidates  # 若无色彩规则，跳过

    scored = []
    for c in candidates:
        try:
            # 下载缩略图并分析色彩
            async with httpx.AsyncClient() as client:
                resp = await client.get(c["thumbnail"], timeout=10.0)
                resp.raise_for_status()
                image_bytes = resp.content

            result = ColorStyleAnalyzer.analyze(image_bytes)
            dominant_colors = [col.lower() for col in result['dominant_colors']]

            # 计算色彩匹配得分（交集比例）
            match_count = sum(1 for dc in dominant_colors if any(tc in dc for tc in target_colors))
            color_score = match_count / len(dominant_colors) if dominant_colors else 0.0

            c["color_score"] = color_score
            scored.append(c)
        except Exception as e:
            c["color_score"] = 0.0
            scored.append(c)

    # 按色彩得分排序（高分优先）
    scored.sort(key=lambda x: x["color_score"], reverse=True)
    return scored


# -------------------------------
# 5️⃣ 四级精筛：Qwen-VL 多模态验证 + 打分
# -------------------------------
async def validate_with_qwen_vl(
    candidates: List[Dict],
    request: VideoRequest
) -> List[Dict]:
    """
    多模态图文一致性 + 风格初步判断
    返回：仅保留【内容一致】的候选（风格可后续调整）
    """
    verified = []
    client = await get_http_client()

    for c in candidates:
        try:
            resp = await client.get(c["thumbnail"])
            resp.raise_for_status()
            image_base64 = base64.b64encode(resp.content).decode('utf-8')

            prompt = f"""
            请综合判断：
            
            【素材信息】
            - 描述：{c['description']}
            - 声称风格：{c['style']}
            - 标签：{', '.join(c['tags'])}

            【用户需求】
            - 内容：{request.description}
            - 目标风格：{request.category}

            请回答：
            1. 图像内容是否真实反映描述？（如：描述“飞行汽车”，图中是否有飞行的汽车？）
            2. 整体视觉是否与用户需求内容一致？
            3. 当前视觉风格是否接近“{request.category}”？（是/否/部分符合）

            请输出 JSON：
            {{
                "content_consistent": true,
                "style_match": "yes|partial|no",
                "reason": "图像显示飞行汽车，内容一致，但色调偏暖，科技感不足"
            }}
            """

            response = await qwen_client.generate(
                prompt=prompt,
                images=[f"data:image/jpeg;base64,{image_base64}"],
                parse_json=True,
                json_schema={
                    "content_consistent": True,
                    "style_match": "yes",
                    "reason": "ok"
                },
                temperature=0.1
            )

            if not response:
                continue

            # ✅ 仅当内容一致时保留
            if response.get("content_consistent", False):
                # 附加 Qwen 对风格的判断，供后续使用
                c["vl_style_judgment"] = response.get("style_match", "no")
                c["vl_reason"] = response.get("reason", "")
                verified.append(c)

        except Exception as e:
            print(f"[Qwen-VL] 验证失败: {str(e)}")
            continue

    return verified


async def decide_by_style_and_color(
    candidates: List[Dict],
    request: VideoRequest
) -> Optional[VideoResponse]:
    """
    关键变更：
    1. 所有候选视频 → 先剪辑出最佳片段（无论风格）
    2. 再判断是否需要风格迁移
    3. 风格迁移输入为【已剪辑的小片段】→ 降低成本
    """
    candidate = candidates[0]
    target_style = request.category
    required_duration = request.duration

    # ✅ STEP 1: 智能剪辑 —— 先定位最相关内容片段（核心前置步骤）
    try:
        final_clip = await select_best_clip_with_vl(candidate, request)
    except Exception as e:
        print(f"[剪辑] 失败，使用默认片段: {str(e)}")
        # fallback：中间截取 required_duration
        dur = candidate["duration"]
        start_fallback = max(0, (dur - required_duration) / 2)
        end_fallback = start_fallback + required_duration
        final_clip = {
            "in_point": start_fallback,
            "out_point": min(end_fallback, dur),
            "confidence": 0.6
        }

    # 提取剪辑后的 in/out
    in_point = final_clip["in_point"]
    out_point = final_clip["out_point"]
    clip_duration = out_point - in_point

    # 确保不超长
    if clip_duration > required_duration:
        out_point = in_point + required_duration

    # ✅ STEP 2: 获取剪辑后的视频元信息（模拟）
    # 实际中，可生成一个临时剪辑 URL，或由 AI 系统接收 in/out
    # 这里我们仍用原 URL，但传入 in/out
    clip_url = candidate["url"]  # 实际可替换为剪辑后临时 URL

    # ✅ STEP 3: 风格与色彩判断（基于原 candidate + VL 判断）
    vl_style_judge = candidate.get("vl_style_judge", "no")
    color_analysis = await analyze_candidate_color(candidate, target_style)
    color_match = color_analysis.get("match", False)

    # --- 最终决策 ---
    if vl_style_judge == "yes" and color_match:
        # ✅ 风格色彩匹配 → 直接返回剪辑片段
        return VideoResponse(
            url=clip_url,
            thumbnail=candidate["thumbnail"],
            in_point=in_point,
            out_point=out_point,
            match_score=final_clip.get("confidence", 0.8)
        )

    else:
        # ❌ 风格不匹配 → 但只迁移【已剪辑的小片段】！
        return await stylize_video_by_ai(
            video_url=clip_url,           # ✅ 输入是剪辑后的小片段
            target_style=target_style,
            duration=required_duration,
            in_point=in_point,            # ✅ 显式传入剪辑区间
            out_point=out_point
        )
    
async def analyze_candidate_color(candidate: Dict, target_style: str) -> Dict:
    """分析单个候选的色彩是否符合目标风格"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(candidate["thumbnail"])
            result = ColorStyleAnalyzer.analyze(resp.content)

        dominant_colors = [c.lower() for c in result['dominant_colors']]
        style_color_map = {
            "科技感": ["blue", "silver", "cyan", "white"],
            "复古": ["brown", "yellow", "beige", "orange"],
            "赛博朋克": ["pink", "purple", "neon", "magenta"],
            "清新": ["green", "blue", "white", "pastel"]
        }
        target_colors = style_color_map.get(target_style, [])

        match = any(any(tc in dc for tc in target_colors) for dc in dominant_colors)
        return {"match": match, "colors": dominant_colors}
    except:
        return {"match": False}  # 分析失败 → 不匹配
    

# 智能剪辑时段推荐
def split_into_segments(analyzed: Dict, window_sec: int = 5) -> List[Dict]:
    """
    将视频切分为 N 秒窗口，聚合内容特征
    """
    duration = analyzed["duration"]
    frames = analyzed["frames"]
    segments = []

    for start in range(0, int(duration), window_sec):
        end = min(start + window_sec, duration)
        segment_frames = [f for f in frames if start <= f["time"] < end]

        # 聚合特征
        objects = [obj for f in segment_frames for obj in f.get("objects", [])]
        speeches = " ".join([f["speech"] for f in segment_frames if f["speech"]])
        avg_motion = sum(f["motion_score"] for f in segment_frames) / len(segment_frames)
        face_count = sum(1 for f in segment_frames if f.get("faces"))

        segments.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "objects": list(set(objects)),
            "speech": speeches,
            "motion_score": avg_motion,
            "face_count": face_count,
            "key_frame_time": start + avg_motion * (end - start),  # 粗略选关键帧
        })

    return segments

async def score_segments_by_desc(segments: List[Dict], user_desc: str) -> List[Dict]:
    """
    使用 Qwen 判断每个 segment 是否匹配用户描述
    仅用于排序，不淘汰
    """
    scored = []
    for seg in segments:
        prompt = f"""
        请判断以下视频片段是否可能包含描述中的内容：
        【用户需求】{user_desc}
        【片段信息】
        - 时间：{seg['start']:.1f}s - {seg['end']:.1f}s
        - 检测物体：{', '.join(seg['objects'][:5])}
        - 语音内容：{seg['speech'][:100]}
        - 动作强度：{seg['motion_score']:.2f}
        - 是否有人脸：{'是' if seg['face_count'] > 0 else '否'}

        请输出 JSON：{{"relevance_score": 0.8}}
        """
        resp = await qwen_generate(prompt, parse_json=True)
        score = resp.get("relevance_score", 0.0) if resp else 0.0
        scored.append({**seg, "relevance_score": score})

    # 按相关性排序
    return sorted(scored, key=lambda x: x["relevance_score"], reverse=True)

async def select_best_clip_with_vl(
    candidate: Dict,           # 候选视频（含 url, description）
    request: VideoRequest,
    top_k: int = 3             # 最多验证 3 个候选片段
) -> Dict:
    """
    为主视频候选选择最佳剪辑区间
    步骤：
    1. 分析视频内容
    2. 切分窗口 + 打分
    3. 对 top-k 片段的关键帧调用 Qwen-VL 验证
    4. 返回最佳 in/out
    """
    video_url = candidate["url"]
    user_desc = request.description

    # Step 1: 视频分析
    analyzed = await analyze_video_content(video_url)
    if not analyzed:
        return {"in_point": 0.0, "out_point": min(10.0, analyzed["duration"]), "vl_verified": False}

    # Step 2: 切分窗口
    segments = split_into_segments(analyzed, window_sec=5)

    # Step 3: 文本打分排序（低成本）
    scored_segments = await score_segments_by_desc(segments, user_desc)
    top_candidates = scored_segments[:top_k]

    # Step 4: 对 top-k 关键帧调用 Qwen-VL（高价值点）
    best_seg = None
    best_vl_score = 0.0
    client = await get_http_client()

    for seg in top_candidates:
        try:
            # 下载关键帧图像（模拟：取中间帧截图 URL）
            key_time = (seg["start"] + seg["end"]) / 2
            thumbnail_url = f"{video_url.replace('.mp4', '')}_thumb_{int(key_time)}.jpg"
            resp = await client.get(thumbnail_url)
            if not resp.is_success:
                continue
            image_base64 = base64.b64encode(resp.content).decode('utf-8')

            # 调用 Qwen-VL 验证该片段是否真实符合描述
            prompt = f"""
            请判断该帧图像是否真实体现用户需求：
            【用户需求】{user_desc}
            【片段时间】{seg['start']:.1f}s - {seg['end']:.1f}s
            【检测内容】物体：{', '.join(seg['objects'][:3])}，语音：{seg['speech'][:80]}

            请回答：
            - 图像是否反映描述内容？
            - 是否存在误导？

            输出 JSON：{{"vl_match": true, "confidence": 0.9}}
            """

            vl_resp = await qwen_client.generate(
                prompt=prompt,
                images=[f"data:image/jpeg;base64,{image_base64}"],
                parse_json=True,
                json_schema={"vl_match": True, "confidence": 0.5},
                temperature=0.1
            )

            if vl_resp and vl_resp.get("vl_match", False):
                confidence = vl_resp.get("confidence", 0.5)
                if confidence > best_vl_score:
                    best_vl_score = confidence
                    best_seg = seg

        except Exception as e:
            print(f"[Qwen-VL] 关键帧验证失败: {str(e)}")
            continue

    # Step 5: 返回最佳片段
    if best_seg:
        return {
            "in_point": best_seg["start"],
            "out_point": min(best_seg["end"], request.duration + best_seg["start"]),  # 控制时长
            "vl_verified": True,
            "confidence": best_vl_score
        }

    # fallback：返回最高文本分的片段（不验证）
    fallback = scored_segments[0]
    dur = request.duration
    out = min(fallback["end"], fallback["start"] + dur)
    return {
        "in_point": fallback["start"],
        "out_point": out,
        "vl_verified": False,
        "confidence": fallback["relevance_score"]
    }