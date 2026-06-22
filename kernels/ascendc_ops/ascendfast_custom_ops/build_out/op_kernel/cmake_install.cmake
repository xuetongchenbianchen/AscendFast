# Install script for directory: /models/share/userdata/cb/AscendFast/kernels/ascendc_ops/add_demo/AddDemoStd/op_kernel

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/add_demo/AddDemoStd/build_out")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/customize/op_impl/ai_core/tbe/config/ascend910_93" TYPE FILE FILES "/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/add_demo/AddDemoStd/build_out/op_kernel/tbe/op_info_cfg/ai_core/ascend910_93/aic-ascend910_93-ops-info.json")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/customize/framework/tensorflow" TYPE FILE FILES "/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/add_demo/AddDemoStd/build_out/op_kernel/tbe/op_info_cfg/ai_core/npu_supported_ops.json")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/" TYPE DIRECTORY FILES "/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/add_demo/AddDemoStd/build_out/op_kernel/binary/dynamic/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/customize/op_impl/ai_core/tbe//kernel/ascend910_93/" TYPE DIRECTORY FILES "/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/add_demo/AddDemoStd/build_out/op_kernel/binary/ascend910_93/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/customize/op_impl/ai_core/tbe//kernel/config/" TYPE DIRECTORY FILES "/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/add_demo/AddDemoStd/build_out/op_kernel/binary/config/")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/add_demo/AddDemoStd/build_out/op_kernel/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
