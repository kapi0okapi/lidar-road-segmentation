#!/bin/bash
# One-time setup on a cluster login node: Python venv, project install and work directories.
# Paths come from slurm/cluster.toml (and slurm/cluster.local.toml). Extra arguments are passed on to pip.
#
#   bash scripts/setup_cluster.sh
set -eo pipefail
cd "$(dirname "$0")/.."

if ! type module &>/dev/null; then
    source /etc/profile
fi
PY_MODULE=$(sed -n 's/^python_module *= *"\([^"]*\)".*/\1/p' slurm/cluster.local.toml slurm/cluster.toml 2>/dev/null | head -1)
module purge
module load "$PY_MODULE"
eval "$(python3 slurm/generate.py --shell-env)"

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel
# PyTorch for CUDA 12.6 runs on every GPU generation of the cluster, V100 to H100. The default PyPI
# build (CUDA 13) has no kernels for V100, which would keep jobs off the often idle V100 nodes.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[train]" "$@"

mkdir -p "$WORK/data" "$WORK/runs" "$WORK/.cache"
python3 slurm/generate.py --all > /dev/null

echo "venv : $VENV"
echo "work : $WORK  (datasets in \$WORK/data, runs in \$WORK/runs)"
echo "jobs : python3 slurm/generate.py --list"
