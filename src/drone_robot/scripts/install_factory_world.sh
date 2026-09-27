#!/usr/bin/env bash
# One-time: put the mlherd factory models where Gazebo / gz-sim expect them.
#   export GAZEBO_MODEL_PATH=$HOME/.gazebo/models/factory/models
set -euo pipefail

DEST="${HOME}/.gazebo/models/factory"
REPO_URL="https://github.com/mlherd/Dataset-of-Gazebo-Worlds-Models-and-Maps.git"
OSRF_URL="https://github.com/osrf/gazebo_models.git"
SRC="${TMPDIR:-/tmp}/factory_src"
OSRF="${TMPDIR:-/tmp}/gz_models"

mkdir -p "${DEST}"
if [ ! -d "${SRC}/.git" ]; then
  git clone --filter=blob:none --sparse "${REPO_URL}" "${SRC}"
fi
git -C "${SRC}" sparse-checkout set worlds/factory
rsync -a --delete --exclude coke_can --exclude disk_part --exclude arm_part \
  "${SRC}/worlds/factory/models/" "${DEST}/models/"
cp -f "${SRC}/worlds/factory/factory.model" "${DEST}/factory.model"

# Classic world uses model://disk_part and model://arm_part; the pack names them *_ariac.
ln -sfn disk_part_ariac "${DEST}/models/disk_part"
ln -sfn arm_part_ariac "${DEST}/models/arm_part"

if [ ! -d "${DEST}/models/coke_can" ]; then
  if [ ! -d "${OSRF}/.git" ]; then
    git clone --filter=blob:none --sparse "${OSRF_URL}" "${OSRF}"
  fi
  git -C "${OSRF}" sparse-checkout set coke_can
  rsync -a "${OSRF}/coke_can/" "${DEST}/models/coke_can/"
fi

echo "Installed factory models at ${DEST} ($(du -sh "${DEST}" | cut -f1))"
echo "export GAZEBO_MODEL_PATH=${DEST}/models"
echo "export GZ_SIM_RESOURCE_PATH=${DEST}/models"
