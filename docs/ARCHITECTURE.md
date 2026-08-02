# MSR Architecture (as built)

This document describes the implementation in `src/msr` as it exists in this
repository. It is a description, not a design proposal — the runtime it
describes is already implemented, tested, and experimentally verified (see
[VERIFICATION.md](VERIFICATION.md)).

## Position in the loop

```
HEKB (Field Prior)
        │
        ▼
Meaning Mapper
        │
        ▼
Meaning Space Runtime
      ├────────► NVS-Kernel
      │
      ▼
     CLE
      ▼
     HEKB
```

MSR sits strictly between Meaning Mapper and CLE/NVS-Kernel/HEKB. It imports
no code from any of the three — every crossing is a frozen dataclass value
(`msr.abi`) passed through a `typing.Protocol` port (`msr.ports`).

- **Fast loop** — every step: `MeaningSpaceRuntime` emits a `KernelView` to
  whatever satisfies `KernelPort`.
- **Slow loop, outbound** — on stabilization: a `StabilizedTrajectory` is
  emitted to whatever satisfies `LiftPort`.
- **Slow loop, inbound** — on request: a `FieldPrior` is pulled from whatever
  satisfies `FieldPriorSource` and installed via `set_field_prior`.

`MSRHost` (`msr.host`) is the only component that touches all three ports; it
owns no meaning-space state itself, only routing and the refresh policy.

## Module map

| Module | Responsibility |
|---|---|
| `msr.abi` | The four frozen boundary types: `MeaningMeasurement`, `MeaningState`, `KernelView`, `StabilizedTrajectory` |
| `msr.errors` | `MSRError` and its three subclasses (`FrameMismatch`, `DimensionMismatch`, `NotPositiveDefinite`) |
| `msr.linalg` | Shared SPD-matrix helpers (shape/finiteness checks, Cholesky-based inversion, tuple freezing) |
| `msr.field` | `FieldPrior` / `GaussianWell` — Φ as a sum of Gaussian wells, basin membership by Mahalanobis radius, `stiffness` (curvature bound) |
| `msr.dynamics` | `InformationAssimilator` (measurement fusion + precision decay) and `LangevinFlow` (relaxation on Φ, with the stiffness-substepping guard) |
| `msr.stability` | `StabilizationDetector` — latched dwell detection, one `StabilizedTrajectory` emission per stabilization |
| `msr.runtime` | `MeaningSpaceRuntime` — the stateful engine; one instance owns one frame |
| `msr.ports` | `KernelPort`, `LiftPort`, `FieldPriorSource` — structural protocols |
| `msr.host` | `MSRHost` — wires the runtime to the three ports and tracks `LoopMetrics` |
| `msr.adapters.mapper` | Parses a Meaning Mapper payload (native or annotation-era field names) into a `MeaningMeasurement` |
| `msr.adapters.hekb` | Converts a HEKB `Concept`-shaped object into a `GaussianWell` / `FieldPrior` |
| `msr.reference` | Non-normative in-process stand-ins for CLE/HEKB/NVS-Kernel — used by the test suite and the experiments, not a specification of those components |

## The step, in detail

1. **Assimilation** (information geometry). A measurement `(θ_obs, Σ)` is
   fused into the runtime's `(θ, P)` in information form:
   `P' = P + Σ⁻¹`, `θ' = P'⁻¹(Pθ + Σ⁻¹θ_obs)`.
2. **Decay**. Positional precision relaxes toward `precision_floor` with time
   constant `forgetting_tau`, so old certainty ages out and later
   measurements can still move the state.
3. **Flow** (meaning physics). `θ` relaxes on Φ for the elapsed `dt`:
   `θ ← θ − γ∇Φ(θ)dt + √(2Tdt)ξ`. `γ` is `mobility`; `T` is `temperature`
   (0 by default — fully deterministic). The step is subdivided under
   `LangevinFlow`'s stiffness guard (§ below) so that a sharp field cannot
   make the explicit integrator diverge.
4. **Reporting**. `MeaningState` is computed (`basin_id` via
   `FieldPrior.basin_of`, `potential` via `FieldPrior.potential`), appended to
   history, and fed to `StabilizationDetector`. A `KernelView` is always
   emitted; a `StabilizedTrajectory` is emitted only on the step that latches
   the dwell.

## Stiffness guard

`FieldPrior.stiffness` is `Σᵢ dᵢ·λ_max(Pᵢ)`, an upper bound on `‖∇²Φ‖`
computed once per field (fields are replaced, never mutated, so this is not
recomputed per step). `LangevinFlow.substep_count` uses it to pick how many
substeps a given `(θ, dt)` needs so that `γ·stiffness·sub_dt ≤
stability_factor`, plus a displacement bound (`max_displacement`) for the
far-field case where the gradient is large but curvature is loose. Total
injected noise variance (`2·T·dt`) is preserved regardless of how many
substeps are taken.

## Stabilization latch policy

`MeaningSpaceRuntime.set_field_prior` drops the `StabilizationDetector`'s
latch **only when the new field reclassifies the current position into a
different basin** (or before any measurement has been ingested). A field
update that leaves the current basin unchanged leaves the dwell a valid
statement. This is load-bearing for the slow loop: see
[VERIFICATION.md](VERIFICATION.md) for the runaway this policy fixes.

## What MSR deliberately does not do

- **Measure** — that is Meaning Mapper's arrow, L1→L2.
- **Lift or persist knowledge** — that is CLE's and HEKB's.
- **Control** — that is NVS-Kernel's; MSR only reports `KernelView`.
- **Import neighbour code** — every neighbour is a structural `Protocol`.
