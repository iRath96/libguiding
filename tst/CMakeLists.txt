find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
INCLUDE ( ./imgui/CMakeLists.txt )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

foreach(DEMO IN ITEMS demo-2d demo-5d)
  add_executable(${DEMO} ${DEMO}.cpp)
  set_target_properties(${DEMO} PROPERTIES CXX_STANDARD 20)

  target_include_directories(${DEMO}
    PRIVATE ${OPENGL_INCLUDE_DIR}
  )

  target_link_libraries(${DEMO}
    PRIVATE libguiding
    PRIVATE ${IMGUI_LIBRARIES}
  )
endforeach(DEMO)
