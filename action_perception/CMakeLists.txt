cmake_minimum_required(VERSION 3.0.2)
project(action_perception)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

#-----------------------------------------------------------------------------------------------------------------------

##############################other package###############################
## Other types of packages                                              ##
## The header file and library file paths need to be specified          ##
##########################################################################

#cuda 
SET(Cuda_INCLUDE_DIRS /usr/local/cuda-11.4/include)
#SET(Cuda_LIBRARIES /usr/local/cuda-11.4/lib64)
link_directories(/usr/local/cuda-11.4/lib64)

#tensorrt 
SET(TensorRT_INCLUDE_DIRS /home/zyq/third_part/TensorRT/include)
#SET(TensorRT_LIBRARIES /home/zyq/third_part/TensorRT/lib)
link_directories(/home/zyq/third_part/TensorRT/targets/x86_64-linux-gnu/lib)
#opencv
SET(OpenCV_DIR /home/zyq/third_part/opencv/build)

#-----------------------------------------------------------------------------------------------------------------------

#############################rospackage###################################
## Find catkin macros and libraries                                     ##
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz) ##
## is used, also find other catkin packages                             ##
## Some packages need to specify the xxx.config file path               ##
##########################################################################

##SET(XXX_DIR PATH)

find_package(catkin REQUIRED COMPONENTS
  OpenCV
  action_msg
  action_base
)

#-----------------------------------------------------------------------------------------------------------------------

###########################################################################
## catkin specific configuration                                         ##
## The catkin_package macro generates cmake config files for your package##
## Declare things to be passed to dependent projects                     ##
## INCLUDE_DIRS: uncomment this if your package contains header files    ##
## LIBRARIES: libraries you create in this project that dependent        ##
##             projects also need                                        ##
## CATKIN_DEPENDS: catkin_packages dependent projects also need          ##
## DEPENDS: system dependencies of this project that dependent           ##
##             projects also need                                        ##
###########################################################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES action_perception
  CATKIN_DEPENDS action_msg action_base
#  DEPENDS system_lib
)

#-----------------------------------------------------------------------------------------------------------------------

###############################Include Path##############################
## Add header file path                                                ##
##                                                                     ##
#########################################################################

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Cuda_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

#-----------------------------------------------------------------------------------------------------------------------

#######################################################################
## Add Source File Path                                              ##
#######################################################################
SET(SOURCE_DIR 
    src/detection_2d/yolov6s.cpp
    src/main.cpp
)

#-----------------------------------------------------------------------------------------------------------------------

#######################################################################
##  Make as dynamic library file  before target_link_libraries       ##
#######################################################################

 add_library(${PROJECT_NAME} SHARED
   ${SOURCE_DIR}
)

#-----------------------------------------------------------------------------------------------------------------------

#######################################################################
##  Generate executable file    before target_link_libraries         ##
#######################################################################
# test
add_executable(${PROJECT_NAME}_test ${SOURCE_DIR})

#-----------------------------------------------------------------------------------------------------------------------

########################################################################
## Link the library you want to depend on                             ##
## Catkin package and other package                                   ##
########################################################################

target_link_libraries(${PROJECT_NAME}_test
  action_base
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  libcudart.so
  libnvinfer.so
  libnvonnxparser.so 

)


