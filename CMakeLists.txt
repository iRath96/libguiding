CMAKE_MINIMUM_REQUIRED ( VERSION 3.0 )
PROJECT ( libguiding )

INCLUDE ( CheckCXXSourceCompiles )
INCLUDE ( CheckCXXSourceRuns )

FILE ( GLOB LIBGUIDING_SOURCE RELATIVE ${PROJECT_SOURCE_DIR}
  include/guiding/*
  src/*
)

ADD_LIBRARY (
  libguiding
  ${LIBGUIDING_SOURCE}
)

TARGET_INCLUDE_DIRECTORIES ( libguiding
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${FFTW_INCLUDES}
)

# if you want to use wrapper.h, make sure you use C++17
# SET_TARGET_PROPERTIES ( libguiding PROPERTIES CXX_STANDARD 17 OUTPUT_NAME guiding )

INSTALL ( TARGETS libguiding
  DESTINATION lib
)

# Build tests by default when building only libguiding

IF ( PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME )
  OPTION ( BUILD_TESTS "Build the tests" ON )
ELSE ()
  OPTION ( BUILD_TESTS "Build the tests" OFF )
ENDIF ()

IF ( BUILD_TESTS )
  ADD_SUBDIRECTORY ( tst )
ENDIF ()
