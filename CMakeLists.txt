cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(kmeans-dpctpp LANGUAGES CXX DESCRIPTION "DPC++ implementation following KMean implemtnation in sklearn-numba-dpex")

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
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.10.1.tar.gz
  URL_HASH SHA256=111014b516b625083bef701df7880f78c2243835abdb263065b6b59b960b6bad
)
FetchContent_MakeAvailable(pybind11)

find_package(PythonExtensions REQUIRED)
find_package(Dpctl REQUIRED)
find_package(NumPy REQUIRED)

set(py_module_name _kmeans_dpcpp)
pybind11_add_module(${py_module_name}
    MODULE
    python_api/_kmeans_lloyd.cpp
)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS} src)
target_link_options(${py_module_name} PRIVATE -fsycl-device-code-split=per_kernel)
install(TARGETS ${py_module_name}
  DESTINATION kmeans_dpcpp
)

set(ignoreMe "${SKBUILD}")

