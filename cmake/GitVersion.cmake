# Fetch the git version from any version tags, or using the default if git
# version tags are not available.
function(get_git_version GIT_VERSION_VAR DEFAULT_VERSION)
  execute_process(
    COMMAND git describe --tags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT GIT_VERSION)
    message(WARNING "Failed to get git version, setting to default ${DEFAULT_VERSION}")
    set(${GIT_VERSION_VAR} "${DEFAULT_VERSION}" PARENT_SCOPE)
  else()
    string(REGEX REPLACE "^v" "" VERSION_NUMBER ${GIT_VERSION})
    set(${GIT_VERSION_VAR} "${VERSION_NUMBER}" PARENT_SCOPE)
  endif()
endfunction()
