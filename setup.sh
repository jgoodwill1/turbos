#!/usr/bin/env bash

set -e

ENV_NAME="turbos"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Updating $ENV_NAME"
    conda env update \
        -n "$ENV_NAME" \
        -f env.yml \
        --prune
else
    echo "Creating $ENV_NAME"
    conda env create \
        -f env.yml
fi

conda activate "$ENV_NAME"

pip install -e .

python -m ipykernel install \
    --user \
    --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"

echo "Environment ready."