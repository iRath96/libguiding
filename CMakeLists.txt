CMAKE_MINIMUM_REQUIRED ( VERSION 3.0 )
PROJECT ( libguiding )

INCLUDE ( CheckCXXSourceCompiles )
INCLUDE ( CheckCXXSourceRuns )

#
# Source folders
#

SET ( LIBGUIDING_CXX11_FEATURES
  cxx_auto_type
  cxx_explicit_conversions
  cxx_lambdas
  cxx_nullptr
  cxx_range_for
  cxx_static_assert
)

#
# external dependencies
#

FIND_PACKAGE ( Eigen3 3.3 REQUIRED NO_MODULE )

#
# Main executables
#

FILE ( GLOB LIBGUIDING_SOURCE
  include/guiding/*

  src/*
)

ADD_LIBRARY (
  libguiding
  ${LIBGUIDING_SOURCE}
)

TARGET_INCLUDE_DIRECTORIES ( libguiding
  PUBLIC include
  PRIVATE ${FFTW_INCLUDES}
)

SET_TARGET_PROPERTIES ( libguiding PROPERTIES CXX_STANDARD 17 OUTPUT_NAME guiding )
TARGET_COMPILE_FEATURES ( libguiding PRIVATE ${LIBGUIDING_CXX11_FEATURES} )
TARGET_LINK_LIBRARIES ( libguiding PRIVATE Eigen3::Eigen )

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
