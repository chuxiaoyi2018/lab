cmake_minimum_required(VERSION 3.25.2)
project(cpp_reference)

include_directories(${PROJECT_SOURCE_DIR})

message("Cpp Reference, starting......")


# set
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Debug")

# add_executable(optional_ optional_.cpp)

# add_executable(move_ move_.cpp)

# add_executable(lock_gurad_ lock_gurad_.cpp)

# add_executable(ptr_shared_ptr_ ptr_shared_ptr_.cpp)


file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

foreach(CPP_FILE ${CPP_FILES})
    get_filename_component(MODULE_NAME ${CPP_FILE} NAME_WE)
    add_executable(${MODULE_NAME} ${CPP_FILE})
endforeach()