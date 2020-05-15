file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES true *.h *.cpp)
set(TEST_BINARY guiding)

INCLUDE ( ../../3rdparty/imgui/CMakeLists.txt )

add_executable(${TEST_BINARY} ${TEST_SOURCES})
set_target_properties(${TEST_BINARY} PROPERTIES CXX_STANDARD 17)

target_link_libraries(${TEST_BINARY}
  PRIVATE libguiding
  PRIVATE ${IMGUI_LIBRARIES}
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
