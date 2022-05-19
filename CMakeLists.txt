cmake_minimum_required(VERSION 3.19)
project(gitrmsheath CXX )

function(export_target tgt_name)
  install(TARGETS ${tgt_name} EXPORT ${tgt_name}-target
      RUNTIME DESTINATION bin
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib)
  install(EXPORT ${tgt_name}-target NAMESPACE ${PROJECT_NAME}::
          DESTINATION lib/cmake/${PROJECT_NAME})
  set(${PROJECT_NAME}_EXPORTED_TARGETS
      ${${PROJECT_NAME}_EXPORTED_TARGETS} ${tgt_name} PARENT_SCOPE)
endfunction(export_target)

macro(gitrmsheath_export_lib target headers)
  export_target(${target})
  install(FILES ${headers} DESTINATION include)
endmacro(gitrmsheath_export_lib)


find_package(Kokkos)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory (src)
add_subdirectory (test)
