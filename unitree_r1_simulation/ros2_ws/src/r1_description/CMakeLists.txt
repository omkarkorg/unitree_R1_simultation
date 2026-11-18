cmake_minimum_required(VERSION 3.8)
project(r1_description)

find_package(ament_cmake REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/urdf/r1_brainco.urdf")
  message(FATAL_ERROR "urdf/r1_brainco.urdf is missing: generate it with tools/build_r1_brainco_model.py "
                      "(the ros2 Dockerfile does this automatically)")
endif()

install(DIRECTORY urdf meshes launch rviz
  DESTINATION share/${PROJECT_NAME})

ament_package()
