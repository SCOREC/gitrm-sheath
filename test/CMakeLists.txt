add_executable (GitrmSheath_Demo GitrmSheath_test.cpp)

target_link_libraries (GitrmSheath_Demo gitrmsheath m)

install(TARGETS GitrmSheath_Demo DESTINATION bin)
