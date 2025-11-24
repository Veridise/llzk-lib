add_tablegen(llzk-tblgen LLZK
  DESTINATION "${CMAKE_BINARY_DIR}/bin"
  EXPORT LLZKTargets
  llzk-tblgen.cpp
  OpCAPIGen.cpp
  CommonCAPIGen.cpp
  CommonAttrOrTypeCAPITestGen.cpp
  OpCAPITestGen.cpp
  DialectCAPITestGen.cpp
  AttrCAPIGen.cpp
  AttrCAPITestGen.cpp
  TypeCAPIGen.cpp
  TypeCAPITestGen.cpp
  EnumCAPIGen.cpp
  EnumCAPITestGen.cpp
)

target_link_libraries(llzk-tblgen PRIVATE
  MLIRTblgenLib
  LLVMSupport
  clangBasic
  clangLex
)

mlir_check_all_link_libraries(llzk-tblgen)
