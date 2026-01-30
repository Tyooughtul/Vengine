cmake_minimum_required(VERSION 3.15)
project(mini-milvus VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开启编译器优化和警告
if(MSVC)
    add_compile_options(/W3 /O2)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

find_package(OpenMP REQUIRED)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- Core Library ----
# 包含 src 目录下的头文件
include_directories(src)

# 收集源文件
file(GLOB_RECURSE SOURCES "src/core/*.cpp")

add_library(core ${SOURCES})
target_include_directories(core PUBLIC src)
target_link_libraries(core PUBLIC OpenMP::OpenMP_CXX)

# ---- Tests ----
enable_testing()

# 简单的测试辅助宏
add_test(NAME TestBasic COMMAND test_basic)

add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic PRIVATE core)

add_executable(test_benchmark tests/test_benchmark.cpp)
target_link_libraries(test_benchmark PRIVATE core)

add_executable(test_thread_pool tests/test_thread_pool.cpp)
target_link_libraries(test_thread_pool PRIVATE core)

add_executable(test_wal tests/test_wal.cpp)
target_link_libraries(test_thread_pool PRIVATE core)

add_executable(test_rwlock tests/test_rwlock.cpp)
target_link_libraries(test_rwlock PRIVATE core)