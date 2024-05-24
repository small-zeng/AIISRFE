cmake_minimum_required(VERSION 2.8.3)
project(exploration_manager)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

set(OpenCV_DIR "/home/nesc525/drivers/5/czbbzc/softwares/opencv/opencv-3.4.15/build")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  visualization_msgs
  sensor_msgs
  geometry_msgs
  quadrotor_msgs
  plan_env
  path_searching
  bspline
  bspline_opt
  traj_utils
  active_perception
  lkh_tsp_solver
  plan_manage
  cv_bridge
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES exploration_manager
 CATKIN_DEPENDS plan_env path_searching bspline bspline_opt active_perception traj_utils lkh_tsp_solver plan_manage
#  DEPENDS system_lib
)


set(TORCH_CUDA_ENABLED ON)  # 启用 CUDA 支持
# libtorch
set(Torch_DIR "/home/drivers/5/czbbzc/softwares/libtorch/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)



# # Output the value of catkin_INCLUDE_DIRS
# message("catkin_INCLUDE_DIRS: ${catkin_INCLUDE_DIRS}")
# message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
# message("DCMAKE_PREFIX_PATH: ${DCMAKE_PREFIX_PATH}")
# message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
# message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
# message("Torch_LIBRARIES: ${Torch_LIBRARIES}")
# message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")


# 定义一个整数变量
set(method_index 6)

# 将变量传递给C++源代码
configure_file(include/exploration_manager/fast_exploration_fsm.h ${PROJECT_SOURCE_DIR}/include/generated/fast_exploration_fsm.h)

# 定义路径列表
set(FAST_EXPLORATION_MANAGER_SRC_PATH_LIST
  src/fast_exploration_manager_fuel.cpp
  src/fast_exploration_manager_fuel_star.cpp
  src/fast_exploration_manager_rec.cpp
  src/fast_exploration_manager_woswitch.cpp
  src/fast_exploration_manager_evpp.cpp
  src/fast_exploration_manager_vpp.cpp
  src/fast_exploration_manager.cpp
)
# 使用 list(GET ...) 获取指定元素
list(GET FAST_EXPLORATION_MANAGER_SRC_PATH_LIST ${method_index} SOURCE_FILE)
# 将第一个元素赋值给变量
set(FAST_EXPLORATION_MANAGER_SRC_PATH ${SOURCE_FILE})
# 打印获取的元素路径
message("Fast exploration manager source path : ${FAST_EXPLORATION_MANAGER_SRC_PATH}")

include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

add_executable(exploration_node
  src/exploration_node.cpp 
  src/fast_exploration_fsm.cpp
  ${FAST_EXPLORATION_MANAGER_SRC_PATH}  # 使用变量来引用路径
  include/exploration_manager/json.hpp
  )
target_link_libraries(exploration_node 
  ${catkin_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_CUDA_LIBRARIES}
  libjsoncpp.a
  # -ldw
  )

add_library(exploration_manager
  ${FAST_EXPLORATION_MANAGER_SRC_PATH}  # 使用变量来引用路径
)
target_link_libraries(exploration_manager
  ${catkin_LIBRARIES}
)

add_executable(write_tsp test/write_tsp.cpp)
target_link_libraries(write_tsp
  ${catkin_LIBRARIES} 
)


# add_executable(load_obj test/load_obj.cpp)
# target_link_libraries(load_obj
#   ${catkin_LIBRARIES} ${PCL_LIBRARIES}
# )


add_executable(test_evpp src/test_evpp.cpp )
target_link_libraries(test_evpp
  ${catkin_LIBRARIES}  ${TORCH_LIBRARIES} ${TORCH_CUDA_LIBRARIES}
)

