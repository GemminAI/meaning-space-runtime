# Changelog

## 1.1.0 — 2026-08-04

Implements Phase 0 (documentation) and Phase 1 (core reliability) of the
MSR Strategic Realignment Plan, itself a response to an architecture audit
of `src/msr` against the canonical `RFC-MSR00`–`06` series. Full disposition
of every audit finding: [docs/RFC_ALIGNMENT.md](docs/RFC_ALIGNMENT.md).

### Phase 0 — Documentation realignment

- Resolved the `RFC-MSR01.md` doc-identity collision: this repository's copy
  is now explicitly authoritative for `src/msr`'s verified behavior; the
  canonical `RFCv3_draft/rfc/MSR/RFC-MSR01.md` is annotated as the series'
  Layer 2 (Architecture) overview, with the collision recorded in both
  documents and in `RFC_MSR_SERIES_INDEX.md` §9.
- Added [docs/RFC_ALIGNMENT.md](docs/RFC_ALIGNMENT.md) (terminology table +
  per-finding disposition) and [docs/BOUNDARIES.md](docs/BOUNDARIES.md)
  (Policy Boundary enforcement is NVS-Kernel's responsibility, not MSR's).

### Phase 1 — Core reliability

- **Breaking.** `LangevinFlow.substep_count` (and therefore `.step`, and
  therefore `MeaningSpaceRuntime.ingest`/`.advance`) now raises the new
  `CFLViolation` error when a field's curvature would require more substeps
  than `max_substeps`, instead of silently clamping to it.
- **Breaking.** `KernelView` gained a required `gradient: Vector` field
  (∇Φ(θ) at the reported position), including in `.as_dict()`.
- Added `tests/test_properties.py`: Hypothesis property-based tests covering
  determinism/replay, SPD and information-fusion invariants, the analytic
  field gradient against finite differences, and CFL bound satisfaction /
  fail-closed behavior.
- Preserved as-is, by design: the two-half-step architecture (Bayesian
  assimilation, then field-only Langevin flow); no Policy Boundary
  mechanism added; no Fisher-Rao metric added (both deferred to Phase 2).

### Verified

- 106 tests (up from 95), 100% line and branch coverage, mypy strict clean,
  ruff clean.

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
