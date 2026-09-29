cmake_minimum_required(VERSION 3.8)
project(drone_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  drone_robot/inference_node.py
  drone_robot/gz_bridge_node.py
  drone_robot/planner_node.py
  scripts/spawn_drone.py
  scripts/install_factory_world.sh
  scripts/bench_factory.py
  scripts/run_gazebo_evals.sh
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch model config models worlds
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(test_preprocess test/test_preprocess.py)
  ament_add_pytest_test(test_astar test/test_astar.py)
  ament_add_pytest_test(test_carve_occupancy test/test_carve_occupancy.py)
endif()

ament_package()
