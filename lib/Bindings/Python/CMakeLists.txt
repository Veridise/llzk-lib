add_library(LLZKPythonBindings OBJECT)
target_sources(LLZKPythonBindings PRIVATE LLZKRegistration.cpp)
target_link_libraries(
  LLZKPythonBindings PUBLIC LLZK::AllDialects Python3::Module
                            pybind11::pybind11)
