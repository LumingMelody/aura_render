"""
测试IMS转换器集成到FastAPI的功能

测试完整流程:
1. 模拟VGP输出
2. 调用IMS转换API
3. 验证转换结果
"""

import requests
import json


BASE_URL = "http://localhost:8001"


def test_ims_convert_api():
    """测试IMS转换API"""
    print("\n" + "=" * 60)
    print("测试1: IMS转换API (/api/ims/convert)")
    print("=" * 60)

    # 模拟VGP输出
    vgp_result = {
        "effects_sequence_id": [
            {
                "id": "clip_001",
                "source_url": "oss://my-bucket/video1.mp4",
                "start": 0.0,
                "end": 5.0,
                "duration": 5.0,
                "transition_out": {
                    "type": "cross_dissolve",
                    "duration": 1.0
                },
                "color_filter": {
                    "preset": "cinematic",
                    "intensity": 0.8
                },
                "visual_effects": [
                    {
                        "type": "lens_flare",
                        "name": "镜头光晕"
                    }
                ]
            },
            {
                "id": "clip_002",
                "source_url": "oss://my-bucket/video2.mp4",
                "start": 5.0,
                "end": 10.0,
                "duration": 5.0,
                "transition_out": {
                    "type": "zoom_transition",
                    "duration": 0.8
                },
                "color_filter": {
                    "preset": "vibrant",
                    "intensity": 1.0
                }
            }
        ],
        "text_overlay_track_id": {
            "clips": [
                {
                    "text": "精彩瞬间",
                    "start": 2.0,
                    "duration": 2.0,
                    "position": "top-center",
                    "style": {
                        "color": "#FFFFFF",
                        "stroke": "#000000",
                        "size": 42,
                        "bold": True
                    }
                }
            ]
        }
    }

    # 发送转换请求
    payload = {
        "vgp_result": vgp_result,
        "use_filter_preset": True,
        "output_config": {
            "MediaURL": "oss://my-bucket/output/video.mp4",
            "Width": 1920,
            "Height": 1080,
            "VideoCodec": "H.264",
            "AudioCodec": "AAC"
        }
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/ims/convert",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("\n✅ 转换成功!")
            print("\n转换摘要:")
            print(json.dumps(result["summary"], indent=2, ensure_ascii=False))

            print("\n\nIMS Timeline (前5行):")
            timeline_str = json.dumps(result["timeline"], indent=2, ensure_ascii=False)
            lines = timeline_str.split('\n')[:20]
            print('\n'.join(lines))
            print("...")

            return True
        else:
            print(f"\n❌ 转换失败: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.ConnectionError:
        print("\n❌ 无法连接到服务器，请确保FastAPI服务正在运行:")
        print("   python app.py")
        return False
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        return False


def test_ims_mappings_api():
    """测试获取映射配置API"""
    print("\n" + "=" * 60)
    print("测试2: 获取IMS映射配置 (/api/ims/mappings)")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/api/ims/mappings", timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("\n✅ 获取映射成功!")

            print("\n转场映射 (部分):")
            transitions = list(result["transitions"].items())[:5]
            for vgp_type, ims_type in transitions:
                print(f"  {vgp_type:20s} → {ims_type}")

            print("\n滤镜预设 (部分):")
            filter_presets = list(result["filters"]["presets"].items())[:5]
            for vgp_preset, ims_preset in filter_presets:
                print(f"  {vgp_preset:20s} → {ims_preset}")

            print("\n特效映射 (部分):")
            effects = list(result["effects"]["mapping"].items())[:5]
            for vgp_effect, ims_effect in effects:
                ims_effect_str = ims_effect if ims_effect else "(不支持)"
                print(f"  {vgp_effect:20s} → {ims_effect_str}")

            return True
        else:
            print(f"\n❌ 获取失败: {response.status_code}")
            return False

    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        return False


def test_ims_preview_api():
    """测试IMS转换预览API"""
    print("\n" + "=" * 60)
    print("测试3: IMS转换预览 (/api/ims/preview)")
    print("=" * 60)

    vgp_result = {
        "filter_sequence_id": [
            {
                "source_url": "oss://bucket/video1.mp4",
                "start": 0.0,
                "end": 5.0,
                "transition_out": {"type": "fade_in_out", "duration": 1.0},
                "color_filter": {"preset": "cinematic"}
            },
            {
                "source_url": "oss://bucket/video2.mp4",
                "start": 5.0,
                "end": 10.0,
                "transition_out": {"type": "cross_dissolve", "duration": 1.2},
                "color_filter": {"preset": "vibrant"}
            }
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/ims/preview",
            json={"vgp_result": vgp_result},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("\n✅ 预览成功!")
            print("\n转换摘要:")
            print(json.dumps(result["summary"], indent=2, ensure_ascii=False))

            print("\n推荐配置:")
            print(json.dumps(result["recommendations"], indent=2, ensure_ascii=False))

            return True
        else:
            print(f"\n❌ 预览失败: {response.status_code}")
            return False

    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        return False


def test_health_check():
    """测试服务健康检查"""
    print("\n" + "=" * 60)
    print("测试0: 健康检查 (/health)")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ 服务正常运行")
            return True
        else:
            print(f"❌ 服务异常: {response.status_code}")
            return False
    except:
        print("❌ 无法连接到服务器")
        return False


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("*  IMS转换器集成测试  *".center(60))
    print("*" * 60)
    print("\n提示: 请确保FastAPI服务正在运行 (python app.py)")
    print("\n")

    # 运行测试
    tests = [
        ("健���检查", test_health_check),
        ("IMS转换API", test_ims_convert_api),
        ("映射配置API", test_ims_mappings_api),
        ("转换预览API", test_ims_preview_api)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试异常: {e}")
            results.append((name, False))

    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s}: {status}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过!")
    else:
        print(f"\n⚠️ {total - passed} 个测试失败")
