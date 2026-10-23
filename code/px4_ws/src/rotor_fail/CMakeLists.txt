cmake_minimum_required(VERSION 3.8)
project(rotor_fail)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Gazebo fault-injection system plugin (loaded via GZ_SIM_SYSTEM_PLUGIN_PATH = <prefix>/lib)
find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED COMPONENTS register)
add_library(RotorFaultPlugin SHARED src/RotorFaultSystem.cpp)
target_link_libraries(RotorFaultPlugin gz-sim::gz-sim gz-plugin::register)
install(TARGETS RotorFaultPlugin LIBRARY DESTINATION lib)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RotorFailure.msg"
  DEPENDENCIES std_msgs
)

install(
  DIRECTORY ext_romfs models launch worlds config policies
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  src/failure_detection.py
  src/fault_injector.py
  src/pftc_node.py
  src/pftc.py
  src/pos_kf.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
