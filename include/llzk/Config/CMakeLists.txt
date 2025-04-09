# We do empty string comparisons to allow for 0 to be specified, as otherwise
# a "0" value is interpreted as false.
if("${CMAKE_PROJECT_VERSION_MAJOR}" STREQUAL "")
    message(FATAL_ERROR "Project must specify major version!")
endif()

if("${CMAKE_PROJECT_VERSION_MINOR}" STREQUAL "")
    message(FATAL_ERROR "Project must specify minor version!")
endif()

if("${CMAKE_PROJECT_VERSION_PATCH}" STREQUAL "")
    message(FATAL_ERROR "Project must specify patch version!")
endif()

if(NOT "${CMAKE_PROJECT_VERSION_TWEAK}" STREQUAL "")
    message(FATAL_ERROR "\"Tweak\" version is unsupported! Please only use semantic versioning major.minor.patch syntax for version.")
endif()

configure_file(Config.h.in Config.h)
