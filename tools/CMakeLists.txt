# add_subdirectory(llzk-driver)

if(LLZK_BUILD_DEVTOOLS)
  add_subdirectory(llzk-lsp-server)
  add_subdirectory(llzk-opt)
endif()
