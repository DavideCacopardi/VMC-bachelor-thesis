cmake_minimum_required(VERSION 3.25)

project(VMC_bachelor_thesis)

# Set CXX standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if the user has specified a build type via the command line, e.g., (in the
# build-directory)
#
#   $cmake .. -DCMAKE_BUILD_TYPE=Release
#
# if not, set Debug as the default build type.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Add all compiler warnings.
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# Add debugging flag for tools such as gdb and valgrind.
set(CMAKE_CXX_FLAGS_DEBUG "-g")
# Add full compiler optimization when in the release build type.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -fno-math-errno")

# Find packages
find_package(Armadillo REQUIRED)
find_package(NLopt REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# Add directories containing header-files
include_directories(
    .
    Hamiltonians
    InitialStates
    Math
    WaveFunctions
    Solvers
    Samplers
    Particles
    VMCOptimizer
    ${ARMADILLO_INCLUDE_DIRS}
    ${NLOPT_INCLUDE_DIRS}
)

# Add sources from all directories
file(
    GLOB SOURCES "*.cpp"
    GLOB SOURCES "Hamiltonians/*.cpp"
    GLOB SOURCES "InitialStates/*.cpp"
    GLOB SOURCES "Math/*.cpp"
    GLOB SOURCES "Solvers/*.cpp"
    GLOB SOURCES "Samplers/*.cpp"
    GLOB SOURCES "Particles/*.cpp"
    GLOB SOURCES "WaveFunctions/*.cpp"
    GLOB SOURCES "VMCOptimizer/*.cpp"
)


# Create executable called "vmc"
add_executable(vmc ${SOURCES})

target_link_libraries(vmc PRIVATE ${ARMADILLO_LIBRARIES} ${NLOPT_LIBRARIES} ${TORCH_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_options(vmc PRIVATE ${TORCH_CXX_FLAGS} ${CMAKE_CXX_FLAGS})

# find_package(Doxygen)

# if (DOXYGEN_FOUND)
#     add_custom_target(doc
#         COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
#         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#         COMMENT "Generating documentation..."
#         VERBATIM
#     )
    
#     message(STATUS "Doxygen found!")
# else()
#     message(WARNING "Doxygen not found. The 'doc' target will not be available.")
# endif()