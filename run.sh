#!/bin/bash

# Aura Render Quick Start Script
# Provides convenient shortcuts for common operations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Print banner
print_banner() {
    print_colored "$CYAN$BOLD" "╔══════════════════════════════════════════════════════════════╗"
    print_colored "$CYAN$BOLD" "║                    🎬 AURA RENDER 🎬                           ║"
    print_colored "$CYAN$BOLD" "║                  Quick Start Script                          ║"
    print_colored "$CYAN$BOLD" "╚══════════════════════════════════════════════════════════════╝"
    echo
}

# Check Python version
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_colored "$RED" "❌ Python 3 is required but not installed"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        print_colored "$RED" "❌ Python 3.8+ is required (found $python_version)"
        exit 1
    fi
    
    print_colored "$GREEN" "✅ Python $python_version is compatible"
}

# Install dependencies
install_deps() {
    print_colored "$BLUE" "📦 Installing dependencies..."
    
    if [[ ! -f "requirements.txt" ]]; then
        print_colored "$RED" "❌ requirements.txt not found"
        exit 1
    fi
    
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    
    print_colored "$GREEN" "✅ Dependencies installed"
}

# Setup environment
setup_env() {
    print_colored "$BLUE" "🔧 Setting up environment..."
    
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            print_colored "$YELLOW" "⚠️  Created .env from .env.example"
            print_colored "$YELLOW" "   Please edit .env and configure your settings"
        else
            print_colored "$RED" "❌ No .env.example file found"
            exit 1
        fi
    else
        print_colored "$GREEN" "✅ .env file exists"
    fi
    
    # Create required directories
    mkdir -p logs temp uploads output materials
    print_colored "$GREEN" "✅ Required directories created"
}

# Quick setup
quick_setup() {
    print_colored "$PURPLE$BOLD" "🚀 Running quick setup..."
    check_python
    install_deps
    setup_env
    print_colored "$GREEN$BOLD" "✅ Setup completed successfully!"
}

# Start server with basic validation
start_server() {
    print_colored "$GREEN$BOLD" "🚀 Starting Aura Render server..."
    
    # Basic checks
    if [[ ! -f ".env" ]]; then
        print_colored "$RED" "❌ .env file not found. Run: ./run.sh setup"
        exit 1
    fi
    
    if [[ ! -f "app.py" ]]; then
        print_colored "$RED" "❌ app.py not found"
        exit 1
    fi
    
    # Choose startup method
    if [[ "$1" == "enhanced" ]]; then
        print_colored "$BLUE" "🔍 Using enhanced startup with diagnostics..."
        exec python3 startup.py
    elif [[ -f "startup.py" ]]; then
        print_colored "$BLUE" "⚡ Using enhanced startup..."
        exec python3 startup.py
    else
        print_colored "$BLUE" "⚡ Using standard startup..."
        exec python3 start.py
    fi
}

# Development mode
dev_mode() {
    print_colored "$BLUE$BOLD" "🛠️  Starting in development mode..."
    
    # Set development environment
    export ENVIRONMENT=development
    export DEBUG=true
    export RELOAD=true
    
    start_server
}

# Test runner
run_tests() {
    print_colored "$BLUE$BOLD" "🧪 Running tests..."
    
    if ! command -v pytest &> /dev/null; then
        print_colored "$YELLOW" "⚠️  pytest not found, installing..."
        python3 -m pip install pytest pytest-asyncio
    fi
    
    python3 -m pytest "$@"
}

# Check health
check_health() {
    print_colored "$BLUE$BOLD" "🏥 Checking application health..."
    
    if [[ -f "scripts/health_check.py" ]]; then
        python3 scripts/health_check.py
    else
        print_colored "$RED" "❌ Health check script not found"
        exit 1
    fi
}

# Show logs
show_logs() {
    if [[ -f "logs/aura_render.log" ]]; then
        tail -f logs/aura_render.log
    else
        print_colored "$RED" "❌ Log file not found"
        exit 1
    fi
}

# Clean up
cleanup() {
    print_colored "$YELLOW$BOLD" "🧹 Cleaning up..."
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    
    # Remove temporary files
    rm -rf .pytest_cache htmlcov .coverage *.tmp 2>/dev/null || true
    
    # Clean logs if specified
    if [[ "$1" == "logs" ]]; then
        rm -rf logs/*.log 2>/dev/null || true
        print_colored "$GREEN" "✅ Logs cleaned"
    fi
    
    print_colored "$GREEN" "✅ Cleanup completed"
}

# Show usage
show_usage() {
    print_colored "$CYAN$BOLD" "Usage: ./run.sh <command> [options]"
    echo
    print_colored "$BLUE" "Available commands:"
    print_colored "$GREEN" "  setup           - Quick setup (install deps, create .env)"
    print_colored "$GREEN" "  start           - Start server with standard startup"
    print_colored "$GREEN" "  start enhanced  - Start server with enhanced diagnostics"
    print_colored "$GREEN" "  dev             - Start in development mode"
    print_colored "$GREEN" "  test [options]  - Run tests with optional pytest options"
    print_colored "$GREEN" "  health          - Check application health"
    print_colored "$GREEN" "  logs            - Show and follow logs"
    print_colored "$GREEN" "  clean [logs]    - Clean temporary files (and logs if specified)"
    print_colored "$GREEN" "  manage <cmd>    - Use Python management script"
    print_colored "$GREEN" "  help            - Show this help message"
    echo
    print_colored "$YELLOW" "Examples:"
    print_colored "$WHITE" "  ./run.sh setup                    # Initial setup"
    print_colored "$WHITE" "  ./run.sh start                    # Start server"
    print_colored "$WHITE" "  ./run.sh dev                      # Development mode"
    print_colored "$WHITE" "  ./run.sh test -v                  # Run tests verbosely"
    print_colored "$WHITE" "  ./run.sh manage start --daemon    # Use management script"
}

# Main execution
main() {
    print_banner
    
    case "$1" in
        "setup"|"install")
            quick_setup
            ;;
        "start")
            start_server "$2"
            ;;
        "dev"|"develop"|"development")
            dev_mode
            ;;
        "test"|"tests")
            shift
            run_tests "$@"
            ;;
        "health"|"check")
            check_health
            ;;
        "logs"|"log")
            show_logs
            ;;
        "clean"|"cleanup")
            cleanup "$2"
            ;;
        "manage")
            shift
            if [[ -f "scripts/manage.py" ]]; then
                python3 scripts/manage.py "$@"
            else
                print_colored "$RED" "❌ Management script not found"
                exit 1
            fi
            ;;
        "help"|"--help"|"-h"|"")
            show_usage
            ;;
        *)
            print_colored "$RED" "❌ Unknown command: $1"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"