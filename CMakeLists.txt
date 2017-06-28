cmake_minimum_required(VERSION 2.6)
project(handschrifterkennung)

add_executable(handschrifterkennung main.cpp)

install(TARGETS handschrifterkennung RUNTIME DESTINATION bin)
