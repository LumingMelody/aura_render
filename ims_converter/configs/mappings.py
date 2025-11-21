"""
VGP到IMS的参数映射配置

包含转场、滤镜、特效、花字等所有效果类型的映射表
"""

# ============================================
# 1. 转场效果映射 (Transition)
# ============================================

VGP_TO_IMS_TRANSITION = {
    # 基础转场
    "cut": None,  # 硬切不需要转场效果
    "fade_in_out": "fade",  # 渐隐
    "cross_dissolve": "fade",  # 叠化 → 渐隐
    "flash_white": "fadecolor",  # 彩色渐隐
    "match_cut": None,  # 匹配剪辑不需要特效

    # 推拉转场
    "wipe_push": "wiperight",  # 推进式模糊 → 向右擦除
    "zoom_transition": "simplezoom",  # 缩放转场 → 放大消失

    # 滑动转场
    "slide": "wiperight",  # 滑动
    "slide_right": "wiperight",
    "slide_left": "wipeleft",
    "slide_up": "wipeup",
    "slide_down": "wipedown",

    # 阿里云IMS官方支持的60种转场效果
    "directional": "directional",  # 对角切换
    "displacement": "displacement",  # 旋涡
    "windowslice": "windowslice",  # 栅格
    "bowTieVertical": "bowTieVertical",  # 垂直领结
    "bowTieHorizontal": "bowTieHorizontal",  # 水平领结
    "simplezoom": "simplezoom",  # 放大消失
    "linearblur": "linearblur",  # 线性模糊
    "waterdrop": "waterdrop",  # 水滴
    "glitchmemories": "glitchmemories",  # 故障
    "polka": "polka",  # 波点
    "perlin": "perlin",  # 蔓延
    "directionalwarp": "directionalwarp",  # 扭曲旋转
    "bounce_up": "bounce_up",  # 向上弹动
    "bounce_down": "bounce_down",  # 向下弹动
    "wiperight": "wiperight",  # 向右擦除
    "wipeleft": "wipeleft",  # 向左擦除
    "wipedown": "wipedown",  # 向下擦除
    "wipeup": "wipeup",  # 向上擦除
    "morph": "morph",  # 雪花消除
    "colordistance": "colordistance",  # 色彩溶解
    "circlecrop": "circlecrop",  # 圆形遮罩
    "swirl": "swirl",  # 中心旋转
    "dreamy": "dreamy",  # 轻微摇摆
    "gridflip": "gridflip",  # 多格翻转
    "zoomincircles": "zoomincircles",  # 圆形放大
    "radial": "radial",  # 圆形扫描
    "mosaic": "mosaic",  # 相册
    "undulatingburnout": "undulatingburnout",  # 波形放大
    "crosshatch": "crosshatch",  # 线性溶解
    "crazyparametricfun": "crazyparametricfun",  # 太空波纹
    "kaleidoscope": "kaleidoscope",  # 万花筒
    "windowblinds": "windowblinds",  # 百叶窗
    "hexagonalize": "hexagonalize",  # 蜂巢溶解
    "glitchdisplace": "glitchdisplace",  # 故障交替
    "dreamyzoom": "dreamyzoom",  # 炫境
    "doomscreentransition_up": "doomscreentransition_up",  # 齿状上升
    "doomscreentransition_down": "doomscreentransition_down",  # 齿状下落
    "ripple": "ripple",  # 波纹
    "pinwheel": "pinwheel",  # 风车
    "angular": "angular",  # 时钟旋转
    "burn": "burn",  # 燃烧
    "circle": "circle",  # 椭圆遮罩
    "circleopen": "circleopen",  # 椭圆溶解
    "colorphase": "colorphase",  # 色相溶解
    "crosswarp": "crosswarp",  # 隧道扭曲
    "cube": "cube",  # 立方体
    "directionalwipe": "directionalwipe",  # 渐变擦除
    "doorway": "doorway",  # 开幕
    "fade": "fade",  # 渐隐
    "fadecolor": "fadecolor",  # 彩色渐隐
    "fadegrayscale": "fadegrayscale",  # 灰色渐隐
    "flyeye": "flyeye",  # 回忆
    "heart": "heart",  # 爱心遮罩
    "luma": "luma",  # 对角开幕
    "multiplyblend": "multiplyblend",  # 多层混合
    "pixelize": "pixelize",  # 像素溶解
    "polarfunction": "polarfunction",  # 花瓣遮罩
    "randomsquares": "randomsquares",  # 随机方块
    "rotatescalefade": "rotatescalefade",  # 旋转
    "squareswire": "squareswire",  # 方块替换
    "squeeze": "squeeze",  # 向内推入
    "swap": "swap",  # 切入
    "wind": "wind",  # 线形擦除

    # 中文映射（用于前端UI）- 完整映射
    # 基础转场
    "渐隐": "fade",
    "彩色渐隐": "fadecolor",
    "灰色渐隐": "fadegrayscale",
    "淡入淡出": "fade",
    "溶解": "fade",

    # 擦除转场
    "向右擦除": "wiperight",
    "向左擦除": "wipeleft",
    "向上擦除": "wipeup",
    "向下擦除": "wipedown",
    "渐变擦除": "directionalwipe",
    "线形擦除": "wind",
    "擦除": "wiperight",
    "左划": "wipeleft",
    "右划": "wiperight",
    "上划": "wipeup",
    "下划": "wipedown",

    # 缩放旋转
    "放大消失": "simplezoom",
    "中心旋转": "swirl",
    "旋转": "rotatescalefade",
    "风车": "pinwheel",
    "时钟旋转": "angular",
    "圆形放大": "zoomincircles",
    "缩放": "simplezoom",

    # 几何图形
    "椭圆遮罩": "circle",
    "圆形遮罩": "circlecrop",
    "椭圆溶解": "circleopen",
    "爱心遮罩": "heart",
    "立方体": "cube",
    "花瓣遮罩": "polarfunction",

    # 特殊效果
    "故障": "glitchmemories",
    "故障交替": "glitchdisplace",
    "燃烧": "burn",
    "波纹": "ripple",
    "水滴": "waterdrop",
    "炫境": "dreamyzoom",
    "轻微摇摆": "dreamy",

    # 扭曲变形
    "旋涡": "displacement",
    "扭曲旋转": "directionalwarp",
    "隧道扭曲": "crosswarp",
    "蔓延": "perlin",
    "对角切换": "directional",

    # 分割网格
    "栅格": "windowslice",
    "百叶窗": "windowblinds",
    "多格翻转": "gridflip",
    "相册": "mosaic",
    "随机方块": "randomsquares",
    "方块替换": "squareswire",
    "线性溶解": "crosshatch",
    "蜂巢溶解": "hexagonalize",

    # 弹跳推入
    "向上弹动": "bounce_up",
    "向下弹动": "bounce_down",
    "向内推入": "squeeze",
    "切入": "swap",
    "开幕": "doorway",
    "对角开幕": "luma",
    "推入": "squeeze",
    "滑动": "wiperight",

    # 创意转场
    "万花筒": "kaleidoscope",
    "波点": "polka",
    "线性模糊": "linearblur",
    "像素溶解": "pixelize",
    "色彩溶解": "colordistance",
    "色相溶解": "colorphase",
    "雪花消除": "morph",
    "波形放大": "undulatingburnout",
    "太空波纹": "crazyparametricfun",
    "圆形扫描": "radial",
    "回忆": "flyeye",
    "多层混合": "multiplyblend",
    "齿状上升": "doomscreentransition_up",
    "齿状下落": "doomscreentransition_down",
    "垂直领结": "bowTieVertical",
    "水平领结": "bowTieHorizontal",
}

# 转场方向映射 (用于自动判断方向性转场)
TRANSITION_DIRECTION_MAP = {
    "push_right": "wiperight",
    "push_left": "wipeleft",
    "push_up": "wipeup",
    "push_down": "wipedown",
    "bounce_up": "bounce_up",
    "bounce_down": "bounce_down",
}

# ============================================
# 2. 滤镜效果映射 (Filter)
# ============================================

# 预设滤镜映射
VGP_TO_IMS_FILTER_PRESET = {
    # VGP基础预设
    "cinematic": "m1",      # 90s现代胶片-复古
    "vibrant": "pl3",       # 清新-春芽
    "monochrome": "pf11",   # 胶片-灰阶
    "dreamy": "pj4",        # 日系-花雾
    "cyberpunk": "electric", # Unsplash-电子
    "natural": "pl1",        # 清新-暗影
    "warm": "f7",           # 80年代负片-咖啡
    "vintage": "pf5",       # 胶片-反转片

    # 扩展映射
    "retro": "m1",          # 复古 → 90s现代胶片-复古
    "cold": "m3",           # 冷色调 → 90s现代胶片-青色
    "nostalgic": "pf4",     # 怀旧 → 胶片-柯达
    "romantic": "pj3",      # 浪漫 → 日系-午后
    "fresh": "pl4",         # 清新 → 清新-明亮
    "dark": "m5",           # 暗调 → 90s现代胶片-暗红
    "bright": "pl4",        # 明亮 → 清新-明亮
    "soft": "a1",           # 柔和 → 90年代艺术胶片-柔和
    "vivid": "a5",          # 鲜艳 → 90年代艺术胶片-鲜艳
    "matte": "a6",          # 哑光 → 90年代艺术胶片-哑光

    # 中文映射（用于前端UI）
    "复古": "m1",
    "黑白": "pf11",
    "暖色": "f7",
    "冷色": "m3",
    "高对比": "m4",
    "柔和": "a1",
    "鲜艳": "a5",
    "戏剧": "m5",
}

# IMS滤镜分类详细信息
IMS_FILTER_CATEGORIES = {
    "90s_modern": {
        "name": "90年代现代胶片",
        "filters": {
            "m1": "复古",
            "m2": "灰阶",
            "m3": "青色",
            "m4": "蓝调",
            "m5": "暗红",
            "m6": "沉闷",
            "m7": "灰橙",
            "m8": "透明"
        }
    },
    "film": {
        "name": "胶片",
        "filters": {
            "pf1": "高调",
            "pf2": "富士",
            "pf3": "暖调",
            "pf4": "柯达",
            "pf5": "复古",
            "pf6": "反转片",
            "pf7": "红外",
            "pf8": "宝丽来",
            "pf9": "禄莱",
            "pf10": "工业",
            "pf11": "灰阶",
            "pf12": "白阶"
        }
    },
    "infrared": {
        "name": "红外",
        "filters": {
            "pi1": "清透",
            "pi2": "黄昏",
            "pi3": "秋色",
            "pi4": "暗调"
        }
    },
    "fresh": {
        "name": "清新",
        "filters": {
            "pl1": "暗影",
            "pl2": "柔和",
            "pl3": "春芽",
            "pl4": "明亮"
        }
    },
    "japanese": {
        "name": "日系",
        "filters": {
            "pj1": "小森林",
            "pj2": "童年",
            "pj3": "午后",
            "pj4": "花雾"
        }
    },
    "unsplash": {
        "name": "Unsplash",
        "filters": {
            "delta": "Delta",
            "electric": "Electric",
            "faded": "Faded",
            "slowlived": "Slow Lived",
            "tokoyo": "Tokyo",
            "urbex": "Urbex",
            "warm": "Warm"
        }
    },
    "80s_negative": {
        "name": "80年代负片",
        "filters": {
            "f1": "济州岛",
            "f2": "雪山",
            "f3": "布达佩斯",
            "f4": "蓝霜",
            "f5": "尤加利",
            "f6": "老街",
            "f7": "咖啡"
        }
    },
    "travel": {
        "name": "旅行",
        "filters": {
            "pv1": "质感",
            "pv2": "天空色",
            "pv3": "清新",
            "pv4": "雾霭",
            "pv5": "高调",
            "pv6": "黑白"
        }
    },
    "90s_artistic": {
        "name": "90年代艺术胶片",
        "filters": {
            "a1": "柔和",
            "a2": "暗调",
            "a3": "青空",
            "a4": "蓝光",
            "a5": "鲜艳",
            "a6": "哑光"
        }
    }
}

# ============================================
# 3. 特效映射 (VFX)
# ============================================

VGP_TO_IMS_EFFECT = {
    # 光影特效
    "lens_flare": "colorfulradial",    # 镜头光晕 → 彩虹射线
    "particle_sparkle": "meteorshower", # 粒子星光 → 流星雨
    "glow_pulse": "neolighting",        # 脉冲辉光 → 霓虹灯

    # 氛围特效
    "film_grain": "oldtvshine",         # 胶片颗粒 → 老电视闪烁
    "noise": "blackwhitetv",            # 噪点 → 电视噪声
    "vignette": None,                   # 暗角 (用滤镜的dark_corner_ratio实现)

    # 动态特效
    "shake": "jitter",                  # 抖动
    "flash": "white",                   # 闪白
    "heartbeat": "heartbeat",           # 心跳

    # 自然特效
    "rain": "rainy",                    # 下雨
    "snow": "snow",                     # 下雪
    "water_ripple": "waterripple",      # 水波
    "lightning": "stormlaser",          # 闪电

    # 梦幻特效
    "fireworks": "fireworks",           # 烟花
    "stars": "colorfulstarry",          # 星空

    # 变形特效
    "fisheye": "fisheye",               # 鱼眼
    "mosaic": "mosaic_rect",            # 马赛克
    "blur": "blur",                     # 模糊

    # 不支持的特效
    "border_glow": None,                # IMS无边缘辉光
    "animated_textbox": None,           # 用文字轨道实现

    # 中文映射（用于前端UI）- 完整映射
    # 基础特效
    "打开": "open",
    "关闭": "close",
    "模糊": "blur",
    "缩放": "zoominout",
    "平移": "open",
    "旋转": "open",

    # 氛围特效
    "彩虹射线": "colorfulradial",
    "流星雨": "meteorshower",
    "星空": "colorfulstarry",
    "发光": "colorfulradial",
    "粒子": "meteorshower",

    # 动态特效
    "闪白": "white",
    "抖动": "jitter",
    "心跳": "heartbeat",
    "霓虹灯": "neolighting",
    "闪光": "white",

    # 光影投射
    "月光投射": "moon_projection",
    "星光投射": "star_projection",
    "爱心投射": "heart_projection",

    # 复古怀旧
    "电视噪声": "blackwhitetv",
    "老电视闪烁": "oldtvshine",
    "夜视仪": "nightvision",
    "故障": "glitchmemories",

    # 梦幻特效
    "烟花": "fireworks",
    "彩色太阳": "colorfulsun",
    "爱心环绕": "heartsurround",

    # 自然特效
    "下雨": "rainy",
    "下雪": "snow",
    "水波": "waterripple",
    "闪电": "stormlaser",

    # 分屏特效
    "2分屏": "splitstill2",
    "9分屏": "splitstill9",
    "跑马灯": "marquee",

    # 色彩特效
    "彩色": "colorful",
    "彩虹滤镜": "rainbowfilter",
    "迪斯科灯光": "discolights",

    # 变形特效
    "鱼眼": "fisheye",
    "马赛克": "mosaic_rect",
    "玻璃": "glass",

    # 其他
    "锐化": None,
    "慢动作": None,
    "倒放": None,
}

# IMS特效分类
IMS_EFFECT_CATEGORIES = {
    "basic": ["open", "close", "blur", "zoominout"],
    "atmosphere": ["colorfulradial", "meteorshower", "colorfulstarry"],
    "dynamic": ["white", "jitter", "heartbeat", "neolighting"],
    "light": ["moon_projection", "star_projection", "heart_projection"],
    "retro": ["blackwhitetv", "oldtvshine", "nightvision"],
    "fantasy": ["fireworks", "colorfulsun", "heartsurround"],
    "nature": ["rainy", "snow", "waterripple", "stormlaser"],
    "splitscreen": ["splitstill2", "splitstill9", "marquee"],
    "color": ["colorful", "rainbowfilter", "discolights"],
    "deform": ["fisheye", "mosaic_rect", "glass"]
}

# ============================================
# 4. 花字效果映射 (Flower Text)
# ============================================

# 花字样式选择策略
VGP_TO_IMS_FLOWER_STYLE = {
    # 类型1: CS系列 (推荐，自带多层描边)
    "bold_stroke": "CS0001-000001",     # 粗体+描边
    "bold_clean": "CS0002-000001",      # 粗体干净
    "elegant": "CS0003-000001",         # 优雅
    "modern": "CS0004-000001",          # 现代
    "classic": "CS0005-000001",         # 经典

    # 类型2: 渐变系列
    "white_gradient": "white_grad",     # 白色渐变
    "red_gradient": "red_grad",         # 红色渐变
    "yellow_gradient": "yellow_grad",   # 黄色渐变
    "blue_gradient": "blue_grad",       # 蓝色渐变
    "green_gradient": "green_grad",     # 绿色渐变
    "purple_gradient": "purple_grad",   # 紫色渐变

    # 主题样式
    "neon": "neon_cyan",                # 霓虹
    "fire": "burning_paper",            # 火焰
    "tropical": "tropical_colors",      # 热带
    "gold": "golden_shine",             # 金色
    "tech": "digital_matrix",           # 科技
}

# 根据颜色自动选择花字样式
COLOR_TO_FLOWER_STYLE = {
    # ✅ 使用有效的阿里云IMS花字样式ID（CS系列）
    "#FFFFFF": "CS0001-000001",  # 白色：经典白色花字
    "#FF0000": "CS0001-000007",  # 红色：红色系统花字
    "#FFFF00": "CS0001-000005",  # 黄色：黄色系统花字
    "#0000FF": "CS0001-000014",  # 蓝色：蓝色系统花字
    "#00FF00": "CS0001-000004",  # 绿色：绿色系统花字
    "#FF00FF": "CS0002-000004",  # 紫色：紫色粗体花字
    "#FFD700": "CS0002-000002",  # 金色：金色粗体花字
    "#00FFFF": "CS0002-000009",  # 青色：青色粗体花字
}

# ============================================
# 5. 位置映射
# ============================================

VGP_TO_IMS_POSITION = {
    "top-left": {"X": 0.1, "Y": 0.1},
    "top-center": {"X": 0.5, "Y": 0.1},
    "top-right": {"X": 0.9, "Y": 0.1},
    "center-left": {"X": 0.1, "Y": 0.5},
    "center": {"X": 0.5, "Y": 0.5},
    "center-right": {"X": 0.9, "Y": 0.5},
    "bottom-left": {"X": 0.1, "Y": 0.9},
    "bottom-center": {"X": 0.5, "Y": 0.9},
    "bottom-right": {"X": 0.9, "Y": 0.9},
}

# ============================================
# 6. 滤镜参数转换配置
# ============================================

# VGP色彩参数到IMS ExtParams的映射规则
FILTER_PARAM_CONVERSION = {
    "brightness": {
        "vgp_range": (0.0, 2.0),      # VGP: 倍数 (1.0为基准)
        "ims_range": (-255, 255),     # IMS: 偏移量 (0为基准)
        "vgp_center": 1.0,
        "ims_center": 0
    },
    "contrast": {
        "vgp_range": (0.0, 2.0),
        "ims_range": (-100, 100),
        "vgp_center": 1.0,
        "ims_center": 0
    },
    "saturation": {
        "vgp_range": (0.0, 2.0),
        "ims_range": (-100, 100),
        "vgp_center": 1.0,
        "ims_center": 0
    },
    "temperature": {
        "vgp_range": (-1.0, 1.0),     # VGP: -1(冷)到1(暖)
        "ims_kelvin_range": (1000, 40000),  # IMS: 色温K值
        "ims_ratio_range": (0, 100),        # IMS: 强度比例
        "neutral_kelvin": 6000
    }
}
