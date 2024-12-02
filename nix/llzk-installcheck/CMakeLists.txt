cmake_minimum_required(VERSION 3.20)
project(LLZKInstallCheck)

find_package(LLZK REQUIRED CONFIG)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE LLZK::LLZKDialect)