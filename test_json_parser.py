#!/usr/bin/env python3
"""测试增强的JSON解析功能"""

import sys
sys.path.insert(0, '.')

from video_generate_protocol.prompt_optimizer import VideoPromptOptimizer

# 创建优化器实例
optimizer = VideoPromptOptimizer()

# 测试各种有问题的JSON格式
test_cases = [
    # 1. 标准JSON（应该成功）
    ('标准JSON', '{"name": "test", "value": 123}'),

    # 2. 带注释的JSON
    ('带注释', '''{
        "name": "test",  // 这是注释
        "value": 123
    }'''),

    # 3. 尾部逗号
    ('尾部逗号', '''{
        "name": "test",
        "value": 123,
    }'''),

    # 4. 单引号
    ('单引号', "{'name': 'test', 'value': 123}"),

    # 5. 混合问题
    ('混合问题', '''{
        'name': 'test',  // 注释
        'color_palette': {
            'main': ['blue', 'green'],  // 主色调
            'accent': ['red'],
        }
    }'''),
]

print("🧪 测试增强的JSON解析功能\n")
print("=" * 60)

success_count = 0
for name, json_str in test_cases:
    print(f"\n测试: {name}")
    print(f"输入: {json_str[:50]}...")

    try:
        result = optimizer._parse_json_robust(json_str)
        print(f"✅ 成功: {result}")
        success_count += 1
    except Exception as e:
        print(f"❌ 失败: {e}")

print("\n" + "=" * 60)
print(f"\n📊 测试结果: {success_count}/{len(test_cases)} 通过")

if success_count == len(test_cases):
    print("🎉 所有测试通过！")
    sys.exit(0)
else:
    print("⚠️ 部分测试失败")
    sys.exit(1)
