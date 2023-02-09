add_executable (GitrmSheath_Demo GitrmSheath_test.cpp)

add_executable (GitrmSheathWachspress_Demo GitrmSheathWachspress_test.cpp)

target_link_libraries (GitrmSheath_Demo gitrmsheath m)

target_link_libraries (GitrmSheathWachspress_Demo gitrmsheath m)

install(TARGETS GitrmSheath_Demo DESTINATION bin)

install(TARGETS GitrmSheathWachspress_Demo DESTINATION bin)
