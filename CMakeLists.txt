cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++14)

set(QT_DIR /opt/homebrew/Cellar/qt@5)
project(playback)

find_package(PCL  REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost 1.71 REQUIRED COMPONENTS system filesystem REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


include_directories(/opt/homebrew/Cellar/boost/1.83.0/lib)
set(BOOST_LIBRARY_DIR /opt/homebrew/Cellar/boost/1.83.0/lib)
set(boost_root /opt/homebrew/Cellar/boost/1.83.0) 
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")


add_executable (ukf_highway src/main.cpp src/ukf.cpp src/tools.cpp src/render/render.cpp)
target_link_libraries (ukf_highway ${PCL_LIBRARIES})




