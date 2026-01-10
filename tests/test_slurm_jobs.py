"""Every job in slurm/cluster.toml must render to a valid-looking sbatch script."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("slurm_generate", ROOT / "slurm" / "generate.py")
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)

CONFIG = generate.load_config()
CONFIG["cluster"]["account"] = "test-account"  # the real one lives in cluster.local.toml


@pytest.mark.parametrize("name", sorted(CONFIG["jobs"]))
def test_job_renders(name):
    job = generate.resolve_job(CONFIG, name)
    script = generate.render(CONFIG, name, job)
    assert script.startswith("#!/bin/bash")
    assert f"#SBATCH --job-name={name}" in script
    if job.get("gpus"):
        assert "#SBATCH --partition=GPUQ" in script
        assert "#SBATCH --gres=gpu:" in script
    if job.get("array"):
        assert f"#SBATCH --array=0-{len(job['array']) - 1}" in script
        assert "$ARRAY_ARGS" in job["command"]
