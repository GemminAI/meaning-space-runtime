# Changelog

## 1.0.0 — 2026-08-02

### Added — initial implementation

- ABI (`msr.abi`): `MeaningMeasurement`, `MeaningState`, `KernelView`,
  `StabilizedTrajectory` — the four frozen types crossing every MSR boundary
- Field prior (`msr.field`): Φ as a sum of Gaussian wells, basin membership,
  curvature bound (`FieldPrior.stiffness`)
- Dynamics (`msr.dynamics`): information-form measurement assimilation +
  Langevin flow, with a CFL-style stiffness-substepping guard
- Stabilization detection (`msr.stability`): latched dwell detector emitting
  one `StabilizedTrajectory` per stabilization
- Runtime (`msr.runtime.MeaningSpaceRuntime`) and host wiring
  (`msr.host.MSRHost`) implementing the fast loop (→ NVS-Kernel) and slow loop
  (→ CLE, ← HEKB field prior)
- Structural ports (`msr.ports`): `KernelPort`, `LiftPort`, `FieldPriorSource`
  — MSR imports no neighbour code
- Adapters (`msr.adapters`): Meaning Mapper payload → `MeaningMeasurement`;
  HEKB `Concept` → `FieldPrior`
- Reference stand-ins (`msr.reference`): non-normative in-process CLE/HEKB/
  kernel, for experiments only
- `RFC-MSR01.md`: specification of the validated implementation
- Three experiments (`experiments/exp_msr_00{1,2,3}_*.py`), all deterministic
  and passing — see `docs/VERIFICATION.md`

### Verified

- 95 tests, 100% line and branch coverage, mypy strict clean, ruff clean
- Two defects found and fixed during experimentation: explicit-Euler
  divergence on stiff lifted wells, and a slow-loop stabilization-latch
  runaway — both documented in `docs/VERIFICATION.md` with regression tests
