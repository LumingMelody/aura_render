#!/bin/bash

# Comprehensive test runner script for Aura Render
# Provides different test execution modes and reporting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
COVERAGE=true
VERBOSE=false
PARALLEL=false
REPORT_FORMAT="html"
OUTPUT_DIR="test-results"

# Help function
show_help() {
    echo "Aura Render Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE       Test type: unit|integration|e2e|api|performance|all (default: all)"
    echo "  -c, --coverage        Enable coverage reporting (default: true)"
    echo "  -v, --verbose         Enable verbose output"
    echo "  -p, --parallel        Run tests in parallel"
    echo "  -f, --format FORMAT   Report format: html|xml|json (default: html)"
    echo "  -o, --output DIR      Output directory for reports (default: test-results)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all tests with default settings"
    echo "  $0 -t unit -v                # Run unit tests with verbose output"
    echo "  $0 -t e2e --no-coverage      # Run e2e tests without coverage"
    echo "  $0 -p -f xml                 # Run all tests in parallel with XML output"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -f|--format)
            REPORT_FORMAT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo -e "${BLUE}Aura Render Test Runner${NC}"
echo -e "${BLUE}========================${NC}"
echo "Test Type: $TEST_TYPE"
echo "Coverage: $COVERAGE"
echo "Verbose: $VERBOSE"
echo "Parallel: $PARALLEL"
echo "Report Format: $REPORT_FORMAT"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check Python environment
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check pytest
if ! python -c "import pytest" &> /dev/null; then
    echo -e "${RED}Error: pytest not installed${NC}"
    echo "Install with: pip install pytest"
    exit 1
fi

# Check required packages for different test types
if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "api" || "$TEST_TYPE" == "e2e" ]]; then
    if ! python -c "import httpx" &> /dev/null; then
        echo -e "${YELLOW}Warning: httpx not found - some API tests may fail${NC}"
    fi
fi

if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "performance" ]]; then
    if ! python -c "import psutil" &> /dev/null; then
        echo -e "${YELLOW}Warning: psutil not found - performance tests may fail${NC}"
    fi
fi

# Check Redis for integration tests
if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "integration" ]]; then
    if ! command -v redis-server &> /dev/null; then
        echo -e "${YELLOW}Warning: Redis not found - some integration tests may be skipped${NC}"
    fi
fi

# Check FFmpeg for video processing tests
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}Warning: FFmpeg not found - video processing tests may be skipped${NC}"
fi

echo -e "${GREEN}Dependencies check completed${NC}"
echo ""

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add test type marker
case $TEST_TYPE in
    unit)
        PYTEST_CMD+=" -m unit"
        ;;
    integration)
        PYTEST_CMD+=" -m integration"
        ;;
    e2e)
        PYTEST_CMD+=" -m e2e"
        ;;
    api)
        PYTEST_CMD+=" -m api"
        ;;
    performance)
        PYTEST_CMD+=" -m performance"
        ;;
    all)
        # Run all tests
        ;;
    *)
        echo -e "${RED}Error: Invalid test type: $TEST_TYPE${NC}"
        echo "Valid types: unit, integration, e2e, api, performance, all"
        exit 1
        ;;
esac

# Add coverage options
if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD+=" --cov=. --cov-report=term-missing"
    
    case $REPORT_FORMAT in
        html)
            PYTEST_CMD+=" --cov-report=html:$OUTPUT_DIR/coverage"
            ;;
        xml)
            PYTEST_CMD+=" --cov-report=xml:$OUTPUT_DIR/coverage.xml"
            ;;
        json)
            PYTEST_CMD+=" --cov-report=json:$OUTPUT_DIR/coverage.json"
            ;;
    esac
fi

# Add verbose output
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD+=" -v"
fi

# Add parallel execution
if [[ "$PARALLEL" == true ]]; then
    # Check if pytest-xdist is available
    if python -c "import xdist" &> /dev/null 2>&1; then
        PYTEST_CMD+=" -n auto"
    else
        echo -e "${YELLOW}Warning: pytest-xdist not found - running tests sequentially${NC}"
    fi
fi

# Add JUnit XML output
PYTEST_CMD+=" --junitxml=$OUTPUT_DIR/junit.xml"

# Add timing information
PYTEST_CMD+=" --durations=10"

# Set up environment
export PYTHONPATH="${PYTHONPATH}:."
export TESTING=true

# Start test execution
echo -e "${BLUE}Starting test execution...${NC}"
echo "Command: $PYTEST_CMD"
echo ""

# Run tests
start_time=$(date +%s)

if eval "$PYTEST_CMD"; then
    exit_code=0
    echo -e "${GREEN}✅ Tests completed successfully${NC}"
else
    exit_code=$?
    echo -e "${RED}❌ Tests failed${NC}"
fi

end_time=$(date +%s)
duration=$((end_time - start_time))

# Print summary
echo ""
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}============${NC}"
echo "Duration: ${duration}s"
echo "Exit Code: $exit_code"

if [[ "$COVERAGE" == true ]]; then
    echo "Coverage Report: $OUTPUT_DIR/coverage"
fi

echo "JUnit Report: $OUTPUT_DIR/junit.xml"

# Generate additional reports if requested
if [[ "$REPORT_FORMAT" == "json" && -f "$OUTPUT_DIR/junit.xml" ]]; then
    echo "Converting JUnit XML to JSON..."
    python -c "
import xml.etree.ElementTree as ET
import json
import sys

try:
    tree = ET.parse('$OUTPUT_DIR/junit.xml')
    root = tree.getroot()
    
    result = {
        'name': root.get('name', 'Test Suite'),
        'tests': int(root.get('tests', 0)),
        'failures': int(root.get('failures', 0)),
        'errors': int(root.get('errors', 0)),
        'skipped': int(root.get('skipped', 0)),
        'time': float(root.get('time', 0)),
        'testcases': []
    }
    
    for testcase in root.findall('.//testcase'):
        tc = {
            'name': testcase.get('name'),
            'classname': testcase.get('classname'),
            'time': float(testcase.get('time', 0)),
            'status': 'passed'
        }
        
        if testcase.find('failure') is not None:
            tc['status'] = 'failed'
            tc['failure'] = testcase.find('failure').text
        elif testcase.find('error') is not None:
            tc['status'] = 'error'
            tc['error'] = testcase.find('error').text
        elif testcase.find('skipped') is not None:
            tc['status'] = 'skipped'
            tc['skipped'] = testcase.find('skipped').text
            
        result['testcases'].append(tc)
    
    with open('$OUTPUT_DIR/test-results.json', 'w') as f:
        json.dump(result, f, indent=2)
        
    print('JSON report generated: $OUTPUT_DIR/test-results.json')
    
except Exception as e:
    print(f'Error generating JSON report: {e}')
"
fi

# Print final status
if [[ $exit_code -eq 0 ]]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
else
    echo -e "${RED}💥 Some tests failed. Check the reports for details.${NC}"
fi

exit $exit_code