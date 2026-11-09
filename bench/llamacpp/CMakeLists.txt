cmake_minimum_required(VERSION 3.16)
project(llama-worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(llama REQUIRED)

add_executable(llama-worker worker.cpp)
target_link_libraries(llama-worker PRIVATE llama)
target_compile_options(llama-worker PRIVATE -Wall -Wextra)
