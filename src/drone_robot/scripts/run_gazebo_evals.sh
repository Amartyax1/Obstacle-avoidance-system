#!/usr/bin/env bash
# Interactive five-world Gazebo eval for screen recording.
# Press Enter before each run, then the bench starts. Stack is killed between worlds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WS="${ROOT}"
CSV="${WS}/generalization_results.csv"
PKG_SHARE="${WS}/install/drone_robot/share/drone_robot"
LAUNCH_PID=""

kill_stack() {
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    pkill -TERM -P "${LAUNCH_PID}" 2>/dev/null || true
    kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
    sleep 1
    pkill -KILL -P "${LAUNCH_PID}" 2>/dev/null || true
    kill -KILL "${LAUNCH_PID}" 2>/dev/null || true
  fi
  LAUNCH_PID=""
  pkill -x rviz2 2>/dev/null || true
  pgrep -f 'gz sim .*/drone_robot/worlds/' | awk '{print $1}' | xargs -r kill -KILL 2>/dev/null || true
  pgrep -f 'inference_node.py|planner_node.py|gz_bridge_node.py|spawn_drone.py' \
    | awk '{print $1}' | xargs -r kill -KILL 2>/dev/null || true
  sleep 1
}

trap kill_stack EXIT

source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${WS}/install/setup.bash"

if [[ ! -d "${PKG_SHARE}" ]]; then
  echo "Build first: colcon build --symlink-install --packages-select drone_robot" >&2
  exit 1
fi

: > "${CSV}"
echo "world,success,crash,astar_interventions,time_sec,east_t,west_t,zmin,zmax" > "${CSV}"

for i in 1 2 3 4 5; do
  world="eval_${i}"
  grid="${PKG_SHARE}/config/${world}_grid.npz"
  if [[ ! -f "${grid}" ]]; then
    echo "missing ${grid}; run train/scripts/generate_warehouse.py --eval-suite" >&2
    exit 1
  fi
  kill_stack
  read -r -p "Press Enter to start evaluation ${i} (${world})..."
  ros2 launch drone_robot ros2launch.py \
    world:="${world}" \
    spawn_x:=-5.0 \
    spawn_y:=0.0 \
    use_planner:=true \
    use_rviz:=true \
    loop_waypoints:=false &
  LAUNCH_PID=$!
  echo "waiting for /odom ..."
  for _ in $(seq 1 40); do
    if ros2 topic list 2>/dev/null | grep -q '^/odom$'; then
      break
    fi
    sleep 1
  done
  sleep 2
  python3 "${WS}/src/drone_robot/scripts/bench_factory.py" \
    --grid "${grid}" \
    --world "${world}" \
    --csv "${CSV}" || true
  kill_stack
done

echo "wrote ${CSV}"
cat "${CSV}"
