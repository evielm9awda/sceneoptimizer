cmake_minimum_required(VERSION 3.10)
project(HyperAdvancedPerformanceOptimizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Boost libraries
find_package(Boost REQUIRED COMPONENTS multiprecision math)
if (NOT Boost_FOUND)
    message(FATAL_ERROR "Boost libraries not found!")
endif()

# Find Eigen library
find_package(Eigen3 REQUIRED)
if (NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 library not found!")
endif()

# Find FFTW library
find_package(FFTW REQUIRED)
if (NOT FFTW_FOUND)
    message(FATAL_ERROR "FFTW library not found!")
endif()

# Include directories
include_directories(${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${FFTW_INCLUDE_DIRS})

# Add executable
add_executable(HyperAdvancedPerformanceOptimizer main.cpp)

# Link libraries
target_link_libraries(HyperAdvancedPerformanceOptimizer ${Boost_LIBRARIES} Eigen3::Eigen FFTW::fftw3)
