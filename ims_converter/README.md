# IMS Converter - VGP到阿里云IMS转换器

将VGP (Video Generate Protocol) 的输出转换为阿里云智能媒体服务(IMS) Timeline格式。

## 📁 目录结构

```
ims_converter/
├── __init__.py              # 包初始化
├── converter.py             # 主转换器类
├── utils.py                 # 转换工具函数
└── configs/
    └── mappings.py          # 参数映射配置
```

## 🚀 快速开始

### 基础用法

```python
from ims_converter import IMSConverter

# 创建转换器 (使用预设滤镜模式)
converter = IMSConverter(use_filter_preset=True)

# VGP输出示例
vgp_result = {
    "filter_sequence_id": [
        {
            "source_url": "https://example.com/video1.mp4",
            "start": 0.0,
            "end": 5.0,
            "transition_out": {
                "type": "cross_dissolve",
                "duration": 1.2
            },
            "color_filter": {
                "preset": "cinematic",
                "intensity": 0.8
            }
        }
    ]
}

# 转换为IMS Timeline
ims_timeline = converter.convert(vgp_result)

# 转换为完整的IMS API请求
ims_request = converter.convert_to_ims_request(
    vgp_result,
    output_config={
        "MediaURL": "oss://my-bucket/output.mp4",
        "Width": 1920,
        "Height": 1080
    }
)
```

## 📊 功能支持

| VGP功能 | IMS对应 | 支持度 | 说明 |
|--------|---------|--------|------|
| **转场(Transition)** | Transition | ✅ 95% | 支持60+种转场效果 |
| **滤镜(Filter)** | Filter | ✅ 90% | 支持预设滤镜和精确参数 |
| **特效(VFX)** | VFX | ⚠️ 60% | 支持概念匹配的特效 |
| **花字(Text)** | Subtitle | ✅ 85% | 支持100+种花字样式 |
| **辅助媒体(Overlay)** | VideoTrack | ✅ 80% | 支持图片/视频叠加 |

## 🔧 转换模式

### 1. 预设滤镜模式 (推荐)

```python
converter = IMSConverter(use_filter_preset=True)
```

**优点**: 简单快速，效果稳定
**映射示例**:
- `cinematic` → IMS `m1` (90s现代胶片-复古)
- `vibrant` → IMS `pl3` (清新-春芽)
- `dreamy` → IMS `pj4` (日系-花雾)

### 2. 精确参数模式

```python
converter = IMSConverter(use_filter_preset=False)
```

**优点**: 精确控制色彩参数
**转换规则**:
```python
VGP参数 (倍数制)        → IMS参数 (偏移制)
brightness: 1.3        → brightness: 76
contrast: 1.2          → contrast: 19
saturation: 0.8        → saturation: -19
temperature: 0.3       → kelvin_temperature: 9000
```

## 📖 映射配置

### 转场映射 (部分)

| VGP类型 | IMS SubType | 效果 |
|---------|-------------|------|
| `cross_dissolve` | `fade` | 渐隐 |
| `zoom_transition` | `simplezoom` | 放大消失 |
| `wipe_push` | `wiperight` | 向右擦除 |
| `swirl` | `swirl` | 中心旋转 |
| `burn` | `burn` | 燃烧 |

完整映射见: `ims_converter/configs/mappings.py`

### 滤镜分类

**90年代现代胶片**: m1-m8
**胶片系列**: pf1-pf12
**日系风格**: pj1-pj4
**清新系列**: pl1-pl4
**Unsplash**: delta, electric, faded, warm...

### 特效映射

| VGP特效 | IMS SubType | 说明 |
|---------|-------------|------|
| `lens_flare` | `colorfulradial` | 彩虹射线 |
| `particle_sparkle` | `meteorshower` | 流星雨 |
| `film_grain` | `oldtvshine` | 老电视闪烁 |
| `rain` | `rainy` | 下雨 |
| `snow` | `snow` | 下雪 |

### 花字样式

**CS系列** (自带多层描边):
- `CS0001-000001` - 粗体+描边
- `CS0002-000001` - 粗体干净
- `CS0003-000001` - 优雅

**渐变系列**:
- `white_grad` - 白色渐变
- `red_grad` - 红色渐变
- `yellow_grad` - 黄色渐变
- `golden_shine` - 金色光泽

## 📝 使用示例

### 示例1: 转场+滤镜

```python
vgp_result = {
    "filter_sequence_id": [
        {
            "source_url": "https://example.com/video1.mp4",
            "start": 0.0,
            "end": 5.0,
            "transition_out": {"type": "fade_in_out", "duration": 1.0},
            "color_filter": {"preset": "cinematic"}
        },
        {
            "source_url": "https://example.com/video2.mp4",
            "start": 5.0,
            "end": 10.0,
            "transition_out": {"type": "zoom_transition", "duration": 0.8},
            "color_filter": {"preset": "vibrant"}
        }
    ]
}

converter = IMSConverter()
ims_timeline = converter.convert(vgp_result)
```

**输出**:
```json
{
  "VideoTracks": [{
    "VideoTrackClips": [
      {
        "MediaURL": "https://example.com/video1.mp4",
        "Effects": [{"Type": "Transition", "SubType": "fade"}]
      }
    ]
  }],
  "EffectTracks": [{
    "EffectTrackItems": [
      {"Type": "Filter", "SubType": "m1", "TimelineIn": 0.0}
    ]
  }]
}
```

### 示例2: 精确滤镜参数

```python
vgp_result = {
    "filter_sequence_id": [{
        "source_url": "https://example.com/video.mp4",
        "start": 0.0,
        "end": 10.0,
        "color_filter": {
            "preset": "custom",
            "applied_params": {
                "brightness": 1.3,    # 增亮30%
                "contrast": 1.2,      # 增加对比度20%
                "saturation": 0.8,    # 降低饱和度20%
                "temperature": 0.3    # 暖色调
            }
        }
    }]
}

converter = IMSConverter(use_filter_preset=False)
ims_timeline = converter.convert(vgp_result)
```

### 示例3: 花字效果

```python
vgp_result = {
    "text_overlay_track_id": {
        "clips": [{
            "text": "太震撼了!",
            "start": 2.0,
            "duration": 3.0,
            "position": "top-center",
            "style": {
                "color": "#FFFFFF",
                "stroke": "#000000",
                "size": 42,
                "bold": True
            }
        }]
    }
}

converter = IMSConverter()
ims_timeline = converter.convert(vgp_result)
```

### 示例4: 完整转换

```python
# 包含转场、滤镜、特效、花字、辅助媒体的完整示例
vgp_result = {
    "effects_sequence_id": [{
        "source_url": "https://example.com/video.mp4",
        "start": 0.0,
        "end": 10.0,
        "transition_out": {"type": "cross_dissolve", "duration": 1.0},
        "color_filter": {"preset": "cinematic"},
        "visual_effects": [{"type": "lens_flare"}]
    }],
    "text_overlay_track_id": {
        "clips": [{
            "text": "精彩瞬间",
            "start": 3.0,
            "duration": 2.0,
            "style": {"color": "#FFD700", "bold": True}
        }]
    },
    "auxiliary_track_id": {
        "clips": [{
            "file_path": "https://example.com/logo.png",
            "start": 0.0,
            "duration": 10.0,
            "type": "image",
            "position": "bottom-right"
        }]
    }
}

converter = IMSConverter()
ims_request = converter.convert_to_ims_request(vgp_result)

# 获取转换摘要
summary = converter.get_conversion_summary(vgp_result)
print(f"转换了 {summary['total_clips']} 个片段")
print(f"包含 {summary['transitions']} 个转场, {summary['texts']} 个文字")
```

## 🧪 测试

运行测试套件:

```bash
python test_ims_converter.py
```

测试包括:
- ✅ 基础转换 (转场+滤镜)
- ✅ 精确滤镜参数转换
- ✅ 特效转换
- ✅ 花字转换
- ✅ 辅助媒体转换
- ✅ 完整转换 (所有功能组合)

## ⚠️ 已知限制

1. **特效位置控制**: IMS特效是全屏效果，不支持VGP的精确位置控制
2. **自定义LUT**: IMS不支持上传自定义LUT文件，只能用预设或color参数
3. **混合模式**: IMS不支持VGP的blend_mode参数
4. **字体**: 花字使用IMS预设样式，不支持VGP的自定义字体URL
5. **透明度**: 部分元素的opacity参数可能无法传递

## 🔄 参数转换规则

### 色彩参数转换

```python
# VGP → IMS
brightness: 0.0-2.0 (倍数) → -255~255 (偏移)
contrast:   0.0-2.0 (倍数) → -100~100 (偏移)
saturation: 0.0-2.0 (倍数) → -100~100 (偏移)
temperature: -1.0~1.0 (冷暖) → 1000~40000K (色温)
```

### 位置转换

```python
# VGP位置字符串 → IMS坐标 (0.0-1.0)
"top-left"      → {"X": 0.1, "Y": 0.1}
"top-center"    → {"X": 0.5, "Y": 0.1}
"center"        → {"X": 0.5, "Y": 0.5}
"bottom-right"  → {"X": 0.9, "Y": 0.9}
```

## 📚 API参考

### IMSConverter

主转换器类

```python
IMSConverter(use_filter_preset: bool = True)
```

**方法**:

- `convert(vgp_result)` - 转换为IMS Timeline
- `convert_to_ims_request(vgp_result, output_config)` - 转换为完整API请求
- `get_conversion_summary(vgp_result)` - 获取转换摘要

### 工具类

- `TransitionConverter` - 转场转换器
- `FilterConverter` - 滤镜转换器
- `EffectConverter` - 特效转换器
- `FlowerTextConverter` - 花字转换器
- `OverlayConverter` - 叠加媒体转换器

## 📄 License

MIT License

## 👥 Contributors

- VGP Team
- IMS Integration Team

---

**完整映射配置**: `ims_converter/configs/mappings.py`
**测试示例**: `test_ims_converter.py`
