#!/bin/bash
set -e

# Ensure build dependencies are installed
pip install --upgrade pip
pip install --upgrade build twine

# Clean and build
./build.sh

# Check the distribution
twine check dist/*

# Upload to TestPyPI first (recommended for testing)
echo "Uploading to TestPyPI..."
twine upload --repository testpypi dist/* --verbose

echo "Testing installation from TestPyPI..."
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ esp-aves

# If testing is successful, upload to the real PyPI
read -p "Do you want to publish to PyPI? (y/n) " -n 1 -r
echo 
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Uploading to PyPI..."
    twine upload dist/*
fi