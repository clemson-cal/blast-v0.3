cmake_minimum_required(VERSION 3.10)
project(blast VERSION 0.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Place executable in project root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_library(READLINE_LIB readline REQUIRED)
if(NOT READLINE_LIB)
    message(FATAL_ERROR "readline library not found. Install with: brew install readline (macOS) or apt install libreadline-dev (Linux)")
endif()

add_executable(blast src/blast.cpp)
target_include_directories(blast PRIVATE ${CMAKE_SOURCE_DIR}/mist/include)
target_link_libraries(blast PRIVATE ${READLINE_LIB})
