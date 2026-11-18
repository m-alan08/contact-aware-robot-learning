#!/usr/bin/env bash
# H2-001: set up a rented Linux + NVIDIA GPU box for ManiFeel (Isaac Gym / TacSL).
#
# UNTESTED: written on 2026-11-18 without access to a GPU box. Run it step by
# step the first time and record what actually happened in the H2-001 report.
#
# Usage:  WORK=/workspace/h2 bash setup_manifeel_box.sh <step>
#   steps: check | clone | isaacgym | install | data | smoke
# Long steps (install, data) should be started with nohup and their logs polled.
set -euo pipefail

WORK=${WORK:-/workspace/h2}
TASK_DATA=${TASK_DATA:-pih_quan_June06}      # Peg Insertion demonstrations (HF: purdue-mars/manifeel)
# Upstream commits pinned on 2026-11-18 (install.sh itself clones unpinned HEADs).
MANIFEEL_SHA=ebfa9e1784848903f642ba1b5233f42511a6de62
ENVS_SHA=1a61da683cf1485f3307684c740771ea5e842b39       # purdue-mars/manifeel-isaacgymenvs
DP_SHA=5ba07ac6661db573af695b419a7947ecb704690f         # real-stanford/diffusion_policy

mkdir -p "$WORK"; cd "$WORK"
step=${1:-check}

case "$step" in
check)
  # Isaac Gym camera sensors (TacSL tactile rendering) need a graphics-capable
  # driver (Vulkan), not just CUDA compute. Containers that expose only
  # NVIDIA_DRIVER_CAPABILITIES=compute,utility will fail here.
  nvidia-smi
  echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}"
  ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null || echo "no Vulkan ICD files found"
  (command -v vulkaninfo && vulkaninfo --summary) || echo "vulkaninfo not installed"
  lsb_release -a 2>/dev/null || cat /etc/os-release
  df -h "$WORK"; free -g
  ;;
clone)
  [ -d manifeel ] || git clone https://github.com/purdue-mars/manifeel.git
  git -C manifeel checkout -q "$MANIFEEL_SHA"
  [ -d manifeel-isaacgymenvs ] || git clone https://github.com/purdue-mars/manifeel-isaacgymenvs.git
  git -C manifeel-isaacgymenvs checkout -q "$ENVS_SHA"
  [ -d diffusion_policy ] || git clone https://github.com/real-stanford/diffusion_policy.git
  git -C diffusion_policy checkout -q "$DP_SHA"
  ;;
isaacgym)
  # The TacSL Isaac Gym package is distributed by the ManiFeel/TacSL authors via
  # Google Drive. It is a binary download: get the user's explicit go-ahead
  # before running this step. The README and install.sh link DIFFERENT files:
  #   README:     13dFRF9EXpzIWaJF2Z6f7BsuPUGQkPE8v
  #   install.sh: 1FHs1tf3QajvYb11UkLaLcDD9THL-C0G5
  # Record which one was used and its sha256.
  FILE_ID=${FILE_ID:?set FILE_ID to the Google Drive id chosen above}
  python3 -m pip install --user gdown
  python3 -m gdown "$FILE_ID" -O IsaacGym_Preview_TacSL_Package.tar.gz
  sha256sum IsaacGym_Preview_TacSL_Package.tar.gz | tee isaacgym_tacsl.sha256
  tar -xzf IsaacGym_Preview_TacSL_Package.tar.gz
  ;;
install)
  # ManiFeel's installer: Miniforge + python 3.8 env 'manifeel' + Isaac Gym +
  # the three repos above (already cloned, so it reuses them) + pinned deps.
  cd manifeel && bash install.sh
  ;;
data)
  source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate manifeel
  mkdir -p manifeel/data
  python - <<EOF
from huggingface_hub import hf_hub_download
p = hf_hub_download("purdue-mars/manifeel", "data/${TASK_DATA}.zip", repo_type="dataset",
                    local_dir="manifeel/data_zip")
print(p)
EOF
  (cd manifeel/data && unzip -q -o "../data_zip/data/${TASK_DATA}.zip")
  sha256sum "manifeel/data_zip/data/${TASK_DATA}.zip" | tee "${TASK_DATA}.sha256"
  ;;
smoke)
  source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate manifeel
  export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
  python -c "from isaacgym import gymtorch; import torch; print('gymtorch ok', torch.__version__, torch.cuda.get_device_name(0))"
  # Zarr keys and force-field storage format (raw 3x10x14 vs. visualised) must be
  # confirmed before the forecaster is trained on them.
  python - <<'EOF'
import glob, zarr
for p in glob.glob('manifeel/data/*'):
    try:
        z = zarr.open(p, 'r')
        print(p); print(z.tree())
    except Exception as e:
        print(p, 'not a zarr root:', e)
EOF
  ;;
*)
  echo "unknown step: $step"; exit 2;;
esac
