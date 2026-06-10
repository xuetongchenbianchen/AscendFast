#!/usr/bin/env bash
# 在本仓库运行任何 Ascend/NPU Python 命令前先 source 本文件。
# 它把 CANN 运行时变量、项目 venv、必需的 torch_npu 启动开关一次性配好。

set -euo pipefail

_ascend_env_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ASCENDFAST_REPO_DIR="${ASCENDFAST_REPO_DIR:-$(cd "${_ascend_env_script_dir}/.." && pwd)}"

# CANN toolkit（本机 8.5.0）：提供 NPU 算子库/驱动环境变量。
export ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${ASCEND_SET_ENV}" ]]; then
  echo "Ascend set_env.sh not found: ${ASCEND_SET_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "${ASCEND_SET_ENV}"

# 项目 venv（由 uv 按 pyproject 建在仓库根 .venv）。
export VIRTUAL_ENV="${VIRTUAL_ENV:-${ASCENDFAST_REPO_DIR}/.venv}"
if [[ ! -x "${VIRTUAL_ENV}/bin/python" ]]; then
  echo "Project virtualenv Python not found: ${VIRTUAL_ENV}/bin/python" >&2
  echo "Create it first:  cd ${ASCENDFAST_REPO_DIR} && uv sync" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${VIRTUAL_ENV}/bin:${PATH}"
# 把仓库根放上 PYTHONPATH，便于从任意 cwd 跑管线模块。
export PYTHONPATH="${ASCENDFAST_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# 关闭 torch_npu 后端自动加载：裸 import torch 会自动加载 torch_npu 后端，某些
# torch/torch_npu/CANN 组合下会崩。这里进程级设上；profile_npu._import_torch()
# 另有一道 os.environ.setdefault 兜底（即便没 source 本脚本也生效）。
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
