#!/bin/bash

# Aura Render 测试套件
# 使用方法: ./run_tests.sh [quick|full|nodes]

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo "🧪 Aura Render 测试套件"
    echo ""
    echo "使用方法:"
    echo "  ./run_tests.sh [选项]"
    echo ""
    echo "选项:"
    echo "  quick    快速测试 (默认)"
    echo "  full     完整测试套件"
    echo "  nodes    测试16个核心节点"
    echo "  help     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./run_tests.sh quick      # 快速测试"
    echo "  ./run_tests.sh nodes      # 测试所有节点"
    echo "  ./run_tests.sh full       # 完整测试"
}

# 检查Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 未找到${NC}"
        return 1
    fi
    
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python版本: $python_version${NC}"
    return 0
}

# 快速测试
run_quick_test() {
    echo -e "${BLUE}🚀 运行快速测试...${NC}"
    echo ""
    
    if python3 test_quick_start.py; then
        echo ""
        echo -e "${GREEN}✅ 快速测试通过！${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ 快速测试失败${NC}"
        return 1
    fi
}

# 节点测试
run_nodes_test() {
    echo -e "${BLUE}🎬 测试16个核心节点...${NC}"
    echo ""
    
    if python3 test_all_nodes.py; then
        echo ""
        echo -e "${GREEN}✅ 节点测试通过！${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ 节点测试失败${NC}"
        return 1
    fi
}

# 完整测试套件
run_full_test() {
    echo -e "${BLUE}🔬 运行完整测试套件...${NC}"
    echo ""
    
    passed=0
    total=0
    
    # 1. 快速测试
    echo -e "${YELLOW}1. 快速基础测试${NC}"
    if run_quick_test; then
        ((passed++))
    fi
    ((total++))
    
    echo ""
    
    # 2. 节点测试
    echo -e "${YELLOW}2. 核心节点测试${NC}"
    if run_nodes_test; then
        ((passed++))
    fi
    ((total++))
    
    # 3. API测试 (如果存在)
    if [ -f "test_api.py" ]; then
        echo ""
        echo -e "${YELLOW}3. API集成测试${NC}"
        if python3 test_api.py; then
            echo -e "${GREEN}✅ API测试通过${NC}"
            ((passed++))
        else
            echo -e "${RED}❌ API测试失败${NC}"
        fi
        ((total++))
    fi
    
    # 4. 单元测试 (如果存在pytest)
    if command -v pytest &> /dev/null && [ -d "tests" ]; then
        echo ""
        echo -e "${YELLOW}4. 单元测试${NC}"
        if pytest tests/ -v; then
            echo -e "${GREEN}✅ 单元测试通过${NC}"
            ((passed++))
        else
            echo -e "${RED}❌ 单元测试失败${NC}"
        fi
        ((total++))
    fi
    
    # 测试结果
    echo ""
    echo "=" * 50
    echo -e "${BLUE}📊 完整测试结果${NC}"
    echo "=" * 50
    echo -e "通过: ${GREEN}$passed${NC}/$total"
    
    if [ $passed -eq $total ]; then
        echo -e "${GREEN}🎉 所有测试通过！${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️ 有部分测试失败${NC}"
        return 1
    fi
}

# 主函数
main() {
    # 检查是否在正确的目录
    if [ ! -f "startup.py" ]; then
        echo -e "${RED}❌ 请在Aura Render项目根目录运行此脚本${NC}"
        exit 1
    fi
    
    # 检查Python环境
    if ! check_python; then
        exit 1
    fi
    
    # 解析参数
    case "${1:-quick}" in
        "quick")
            run_quick_test
            ;;
        "nodes")
            run_nodes_test
            ;;
        "full")
            run_full_test
            ;;
        "help"|"-h"|"--help")
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ 未知选项: $1${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"