cmake_minimum_required(VERSION 3.0.2)
project(action_base)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

#-----------------------------------------------------------------------------------------------------------------------
####### cmakeconfig 路径###
### 既包含源文件又包含头文件 ###
### 1.如果没有安装在默认路径下需要用XXX_DIR指明XXXconfig.cmake的路径
### 2.find_package
### 3.include_directories 引入头文件
### 4.target_link_libraries 引入库文件
### 5.在功能包package.xml文件中添加三个依赖


##############################other package###############################
## Other types of packages                                              ##
## The header file and library file paths need to be specified          ##
##########################################################################

set(OpenCV_DIR /home/zyq/third_part/opencv/build)
### 只包含头文件 ###
### 既可以自定义只引入头文件也可以用find_package引入
set(Eigen3_DIR /usr/lib/cmake/eigen3)
###### Third party library installation path###/

#-----------------------------------------------------------------------------------------------------------------------
###### 寻找cmake构建的包 ###
#1.在find_package引入的包 自动给XXX_LIBRARIES 和xxx_INCLUDE_DIRS赋值
#2.finde_package引入的包  会在默认路径和XXX_DIR制定的路径下寻找xxxConfig.cmake文件
#3.ros自带的包不需要再进行引入头文件和链接库操作


#############################rospackage###################################
## Find catkin macros and libraries                                     ##
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz) ##
## is used, also find other catkin packages                             ##
## Some packages need to specify the xxx.config file path               ##
##########################################################################

find_package(catkin REQUIRED COMPONENTS
  OpenCV
  cv_bridge
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
  Eigen3
)

#-----------------------------------------------------------------------------------------------------------------------
###### 传递包之间的依赖关系(系统依赖和Cmake依赖包) ###
### 1.INCLUDE_DIRS 传递给依赖这个包的头文件库
### 2.传递给依赖这个包的功能包一些catkin包
### 3.传递给依赖这个包的功能包一些非catkin包


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
  INCLUDE_DIRS include   ${Eigen3_DIR}
  CATKIN_DEPENDS cv_bridge pcl_conversions pcl_ros roscpp rospy sensor_msgs
  DEPENDS  OpenCV
)

#-----------------------------------------------------------------------------------------------------------------------
### 将头文件目录放到头文件搜素路径中 默认相对路径 ###


###############################Include Path##############################
## Add header file path                                                ##
#########################################################################

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

#-----------------------------------------------------------------------------------------------------------------------
### 库文件的源文件放在这里 ###


#######################################################################
## Add Source File Path                                              ##
#######################################################################
SET(SOURCE_DIR src/test.cpp)

#-----------------------------------------------------------------------------------------------------------------------

#######################################################################
##  Make as dynamic library file  before target_link_libraries       ##
#######################################################################

### 将该文件编译成库   ###
 add_library(${PROJECT_NAME} SHARED
   ${SOURCE_DIR}
)

#-----------------------------------------------------------------------------------------------------------------------6
# 添加可执行文件，注意链接的对象要对应

#######################################################################
##  Generate executable file    before target_link_libraries         ##
#######################################################################
# test
#add_executable(${PROJECT_NAME}_test ${SOURCE_DIR})

#-----------------------------------------------------------------------------------------------------------------------7
### 链接目标和需要的库 ##

########################################################################
## Link the library you want to depend on                             ##
## Catkin package and other package                                   ##
########################################################################

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
