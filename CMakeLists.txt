cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()

project(kmeans-dpctpp LANGUAGES CXX DESCRIPTION "DPC++ implementation following KMean implemtnation in sklearn-numba-dpex")

option(CUDA "Whether to compiler for nvidia-nvptx-cuda target" OFF)

find_package(IntelDPCPP REQUIRED)

if (NOT DEFINED DPCTL_MODULE_PATH)
  if (DEFINED ENV{DPCTL_MODULE_PATH})
    set(DPCTL_MODULE_PATH $ENV{DPCTL_MODULE_PATH})
  else ()
    message(FATAL_ERROR "Specify DPCTL_MODULE_PATH, either via cmake or as environment varibale")
  endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
  URL_HASH SHA256=d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c
)
FetchContent_MakeAvailable(pybind11)

find_package(Python REQUIRED COMPONENTS Development.Module NumPy)
find_package(Dpctl REQUIRED)

set(py_module_name _kmeans_dpcpp)
pybind11_add_module(${py_module_name}
    MODULE
    python_api/_kmeans_lloyd.cpp
)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS} src)
if (CUDA)
  target_compile_options(${py_module_name} PUBLIC -fsycl-targets=nvptx64-nvidia-cuda,spir64-unknown-unknown)
endif()
target_link_options(${py_module_name} PRIVATE -fsycl-device-code-split=per_kernel)
install(TARGETS ${py_module_name}
  DESTINATION kmeans_dpcpp
)

set(ignoreMe "${SKBUILD}")

