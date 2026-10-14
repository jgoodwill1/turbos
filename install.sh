set -e

ENV_NAME="turbos"

conda activate "$ENV_NAME"

pip install -e .

python -m ipykernel install \
    --user \
    --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"