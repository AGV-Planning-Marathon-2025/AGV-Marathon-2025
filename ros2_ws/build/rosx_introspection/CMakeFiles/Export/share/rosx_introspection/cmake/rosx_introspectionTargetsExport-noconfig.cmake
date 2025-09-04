#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "rosx_introspection::rosx_introspection" for configuration ""
set_property(TARGET rosx_introspection::rosx_introspection APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(rosx_introspection::rosx_introspection PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/librosx_introspection.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS rosx_introspection::rosx_introspection )
list(APPEND _IMPORT_CHECK_FILES_FOR_rosx_introspection::rosx_introspection "${_IMPORT_PREFIX}/lib/librosx_introspection.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
