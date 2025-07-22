#ifndef MLIR_DIALECT_QUANTUM_IR_QUANTUMTRAITS_H
#define MLIR_DIALECT_QUANTUM_IR_QUANTUMTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace quantum {

/// Marker trait for self-inverse gates
template <typename ConcreteType>
struct SelfInverse : public OpTrait::TraitBase<ConcreteType, SelfInverse> {};

/// Marker trait for Hermitian gates
template <typename ConcreteType>
struct Hermitian : public OpTrait::TraitBase<ConcreteType, Hermitian> {};

} // namespace quantum
} // namespace mlir

#endif // MLIR_DIALECT_QUANTUM_IR_QUANTUMTRAITS_H
