import mlir.ir
from . import registration

def add_dialects_to_context(context: "mlir.ir.Context"):
    """Registers all LLZK dialects in the context."""
    registry = mlir.ir.DialectRegistry()
    registration.register_dialects(registry)
    context.append_dialect_registry(registry)
