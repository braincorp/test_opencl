project(test_opencl)
set(PROJECT_NAME test_opencl)
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

## Both Boost.python and Python libs are required.
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs 2.7 REQUIRED)

# Find actual numpy in use (as opposed to system numpy)
execute_process(
COMMAND "python2.7" -c
        "from __future__ import print_function\ntry: import numpy; print(numpy.__version__, end='')\nexcept:pass\n"
OUTPUT_VARIABLE Numpy_VERSION)
set( REQUIRED_NUMPY_VERSION "1.11.2" )
if( NOT (${Numpy_VERSION} STREQUAL ${REQUIRED_NUMPY_VERSION}) )
  message( FATAL_ERROR "Bad numpy version ${Numpy_VERSION}. Please run 'sudo pip install numpy==${REQUIRED_NUMPY_VERSION}'" )
endif()
execute_process(
COMMAND "python2.7" -c
        "from __future__ import print_function\ntry: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
        OUTPUT_VARIABLE Numpy_INCLUDE_DIRS)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${Numpy_INCLUDE_DIRS}  # this should come before PYTHON includes because it also contains system numpy
        ${PYTHON_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

##############################################################################
# Define package
##############################################################################
catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  DEPENDS OpenCV
)

##############################################################################
# Build
##############################################################################

# libraries

add_library(test_opencl_module SHARED src/voxel_raytrace_gpu_module.cpp src/gpu_utils.cpp)

target_link_libraries(test_opencl_module ${Boost_LIBRARIES} OpenCL)

# Don't prepend wrapper library name with lib and add to Python libs.
set_target_properties(test_opencl_module PROPERTIES
        PREFIX "_wrapper_"
        LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
        )
