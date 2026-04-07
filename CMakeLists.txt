cmake_minimum_required(VERSION 3.16)
project(gpt2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -ffast-math)
endif()

find_package(OpenMP REQUIRED)

add_library(gpt2_lib STATIC
    src/ops_naive.cpp
    src/model.cpp
    src/tokenizer.cpp
)
target_include_directories(gpt2_lib PUBLIC include)
target_link_libraries(gpt2_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(gpt2 src/main.cpp)
target_link_libraries(gpt2 PRIVATE gpt2_lib)