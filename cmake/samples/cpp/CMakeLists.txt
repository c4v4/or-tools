# Copyright 2010-2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.14)
project(Sample VERSION 1.0.0 LANGUAGES CXX)

include(CTest)
find_package(ortools CONFIG REQUIRED)

add_executable(sample main.cpp)
target_compile_features(sample PUBLIC cxx_std_17)
set_target_properties(sample PROPERTIES VERSION ${PROJECT_VERSION})
target_link_libraries(sample PRIVATE ortools::ortools)

if(BUILD_TESTING)
  add_test(NAME sample_UT COMMAND sample)
endif()

include(GNUInstallDirs)
install(TARGETS sample
  EXPORT SampleTargets
  DESTINATION ${CMAKE_INSTALL_BINDIR})
