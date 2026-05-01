#!/bin/bash
# Set up the conda environment from environment.yml.

ENV_FILE="environment.yml"
ENV_NAME=$(sed -n 's/^name:[[:space:]]*//p' "$ENV_FILE" | head -n 1)

if [ -z "$ENV_NAME" ]; then
    echo "Could not read environment name from $ENV_FILE."
    exit 1
fi

cleanup_build_artifacts() {
    if [ -d "build" ]; then
        echo "Removing local build artifacts..."
        rm -rf build
    fi
}

echo "Setting up AutoREC environment..."
echo ""

# Check if environment already exists
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "$ENV_NAME" || exit 1
    else
        echo "Setup cancelled."
        exit 0
    fi
fi

# Create environment from environment.yml. The pip section installs this package
# from pyproject.toml, including the notebook extra.
echo "Creating conda environment from environment.yml..."
conda env create -f "$ENV_FILE" || exit 1
cleanup_build_artifacts

echo ""
echo "Setup complete."
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
