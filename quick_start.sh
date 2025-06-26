#!/bin/bash

# AutoSAD Quick Start Script
# This script helps you get started with AutoSAD experiments quickly

echo "AutoSAD Quick Start"
echo "==================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

echo "Python found: $(python --version)"

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import numpy, scipy, sklearn, psutil, pysad" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Some dependencies are missing. Installing from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
else
    echo "All dependencies are available"
fi

# Create results directory
mkdir -p benchmark_results
echo "Created benchmark_results directory"

echo ""
echo "Quick Start Examples:"
echo "===================="

echo ""
echo "1. Run AutoSAD on a small dataset:"
echo "   python scripts/autosad_run.py --dataset 15_Hepatitis"

echo ""
echo "2. Run all models on a specific dataset:"
echo "   python run_scripts.py --mode dataset --name 5_campaign"

echo ""
echo "3. Run AutoSAD on all datasets:"
echo "   python run_scripts.py --mode model --name autosad"

echo ""
echo "4. Quick test (AutoSAD on Hepatitis dataset):"
read -p "   Would you like to run this test now? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running quick test..."
    python scripts/autosad_run.py --dataset 15_Hepatitis --progress_interval 100
    echo ""
    echo "Test completed! Check benchmark_results/ for output."
fi

echo ""
echo "For more options, see the README.md file."
echo "Happy experimenting!"
