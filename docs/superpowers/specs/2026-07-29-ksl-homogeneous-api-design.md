# Homogeneous KSL API Design

## Goal

Restrict `tt_ksl` to the homogeneous equation

```text
dx/dt + A*x = 0
```

and replace the ambiguous integer `passes` option with
`symmetric::Bool=true`.

## Public API

The sole supported method is:

```julia
tt_ksl(
    x0::TTTensor,
    A::TTMatrix,
    tau::Real;
    tol::Real=1e-12,
    rmax::Int=typemax(Int),
    symmetric::Bool=true,
    expmethod::Symbol=:krylov,
    krylov_m::Int=30,
) -> TTTensor
```

The method `tt_ksl(x0, A, y, tau; ...)` is removed rather than deprecated.
The `passes` keyword is removed rather than deprecated.

## Sweep Semantics

With `symmetric=true`, the integrator performs a left-to-right half-sweep
using `tau/2`, followed by a right-to-left half-sweep using `tau/2`. This is
the default symmetric KSL scheme.

With `symmetric=false`, the integrator performs only the left-to-right sweep
using the full `tau`. This is the nonsymmetric KSL scheme.

No integer sweep count is accepted. In particular, the implementation cannot
silently integrate a fraction of the requested interval through unsupported
values such as the former `passes=3`.

## Internal Structure

The homogeneous implementation operates directly on the initial TT cores and
operator environments. It does not construct a zero right-hand-side tensor.

All nonhomogeneous state and operations are removed:

- the source tensor argument and converted source cores;
- the `phiy` environments;
- projected RHS calculation in both sweeps;
- source additions and subtractions in the core and bond updates;
- `compute_next_Phi_y`;
- `compute_rhs_ksl`;
- the homogeneous convenience wrapper that currently delegates through a
  zero source tensor.

The existing operator projection, local matrix exponential, SVD/QR updates,
rank cap, real Krylov path, and complex Taylor path remain unchanged unless a
test exposes a homogeneous KSL defect.

## Compatibility

This is an intentional breaking API change. Repository search found no caller
of the nonhomogeneous overload or the `passes` keyword. The existing KSL
example already calls `tt_ksl(x0, A, tau; ...)` and therefore requires no
behavioral change.

The docstring describes only the homogeneous equation and documents
`symmetric`.

## Tests

Automated tests cover observable homogeneous behavior:

1. A one-site real problem agrees with `exp(-tau*A)*x0` for both
   `symmetric=false` and `symmetric=true`.
2. A full-rank two-site complex problem agrees with the dense matrix
   exponential under the symmetric scheme.
3. A multi-site unitary evolution preserves the dense-vector norm within a
   numerical tolerance.
4. Calling the removed nonhomogeneous signature produces `MethodError`.
5. Passing the removed `passes` keyword produces an unsupported-keyword
   error.

The complete existing test suite must remain green.

## Non-goals

- Supporting time-dependent operators.
- Supporting nonhomogeneous/source terms.
- Adding rank adaptation or enrichment.
- Changing the exponential approximation algorithms.
- Reproducing the Fortran routine's starting sweep direction exactly.
