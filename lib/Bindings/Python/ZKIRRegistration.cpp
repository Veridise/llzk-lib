/// This defines a module that can be used to register the ZKIR dialects.

#include "ZKIR/InitDialects.h"
#include <mlir/Bindings/Python/PybindAdaptors.h>
#include <mlir/CAPI/IR.h>

PYBIND11_MODULE(_zkirRegistration, m) {
  m.doc() = "ZKIR dialect registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    zkir::registerAllDialects(*unwrap(registry));
  });
}
