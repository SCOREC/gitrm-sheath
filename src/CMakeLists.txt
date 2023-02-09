set(HEADERS
  GitrmSheath.hpp
  GitrmSheathUtils.hpp
  GitrmSheathTestUtils.hpp
)

set(SOURCES
  GitrmSheath.cpp
  GitrmSheathTestUtils.cpp
)

add_library (gitrmsheath-core ${SOURCES})

target_include_directories(gitrmsheath-core INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include>)
target_link_libraries(gitrmsheath-core Kokkos::kokkos)
target_link_libraries (gitrmsheath-core PkgConfig::NetCDF)


gitrmsheath_export_lib(gitrmsheath-core "${HEADERS}")

add_library (gitrmsheath INTERFACE)
target_link_libraries(gitrmsheath INTERFACE ${gitrmsheath_EXPORTED_TARGETS})
export_target(gitrmsheath)
