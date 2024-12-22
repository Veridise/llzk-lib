/// This defines a module that can be used to register the LLZK dialects.

#include "LLZK/InitDialects.h"

#include <mlir/Bindings/Python/PybindAdaptors.h>
#include <mlir/CAPI/IR.h>

PYBIND11_MODULE(_llzkRegistration, m) {
  m.doc() = "LLZK dialect registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    llzk::registerAllDialects(*unwrap(registry));
  });
}
