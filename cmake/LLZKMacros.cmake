macro(llzk_target_add_mlir_link_settings target)
  llvm_update_compile_flags(${target})
  mlir_check_all_link_libraries(${target})
endmacro()

function(llzk_add_mlir_doc target_name out_filename tblgen_flags)
  # this is a modified version of add_mlir_doc from AddMLIR.cmake
  set(OUT_FILE "${LLZK_MLIR_DOC_OUTPUT_DIR}/${out_filename}")
  tablegen(MLIR ${out_filename} ${tblgen_flags} ${ARGN})
  add_custom_command(
    OUTPUT ${OUT_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_BINARY_DIR}/${out_filename}" "${OUT_FILE}"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${out_filename}")
  add_custom_target(${target_name} DEPENDS ${OUT_FILE})
  add_dependencies(mlir-doc ${target_name})
endfunction()
