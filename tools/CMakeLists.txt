# add_subdirectory(zkir-driver)

if (ZKIR_BUILD_DEVTOOLS)
  add_subdirectory(zkir-lsp-server)
  add_subdirectory(zkir-opt)
endif()
