add_library(ZKIRPythonBindings OBJECT)
target_sources(ZKIRPythonBindings PRIVATE ZKIRRegistration.cpp)
target_link_libraries(ZKIRPythonBindings
  PUBLIC
    ZKIR::AllDialects
    Python3::Module
    pybind11::pybind11
)
