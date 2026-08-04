# Changelog

## 1.1.0 — 2026-08-04

Implements Phase 0 (documentation) and Phase 1 (core reliability) of the
MSR Strategic Realignment Plan, itself a response to an architecture audit
of `src/msr` against the canonical `RFC-MSR00`–`06` series, and adds the
EXP-Ubuntu004 verification surface. Full disposition of every audit finding
and the EXP-Ubuntu004 gap analysis: [docs/RFC_ALIGNMENT.md](docs/RFC_ALIGNMENT.md).

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

### Added — EXP-Ubuntu004 verification surface

Additions needed to compute EXP-Ubuntu004's five verification metrics over
an already-realized trajectory. None of this is normative RFC-MSR01 behavior
— see `docs/RFC_ALIGNMENT.md` for the gap this intentionally leaves open,
and `docs/VERIFICATION.md`/`RFC-MSR01.md` for what remains unchanged and
still verified as of 1.0.0.

- `FieldPrior.hessian` (`msr.field`): analytic ∇²Φ(θ), the closed form
  already documented (but unimplemented) in the `stiffness` docstring.
  Verified against finite differences of the existing `gradient()`.
- `MeaningState.as_dict`/`from_dict`, `StabilizedTrajectory.as_dict`/
  `from_dict` (`msr.abi`): deterministic, numpy-free round-trip
  serialization, extending the existing `KernelView.as_dict` pattern.
- `msr.metrics` (new): `runtime_stability_rate` (finite-time Runtime
  Stability λ = -1/T·ln(‖Σ(T)‖_F/‖Σ(0)‖_F), Frobenius norm chosen and
  documented explicitly) and `drift_distance`/`drift_converged` (pointwise
  distance and windowed-threshold check against an experiment-supplied
  reference trajectory).
- `msr.lyapunov` (new): `largest_lyapunov_exponent`, a Benettin-method
  finite-time largest Lyapunov exponent over the deterministic drift's
  tangent map. Three approximations are documented in the module
  docstring, not hidden: deterministic-backbone-only, single-step Euler per
  *recorded* interval (coarser than the primal flow's own adaptive
  substepping on a stiff field), and a fixed (non-random) initial
  perturbation direction.
- `InsufficientHistory` added to the `MSRError` taxonomy, for "not enough
  samples/elapsed time to measure a rate," distinct from a shape defect.
- `fail_under = 100` added to `[tool.coverage.report]`, matching the
  100%-coverage convention already achieved but not previously enforced by
  config.

### Added — Phase 3: NVS-Kernel basin recovery experiment

`experiments/exp_msr_004_recovery.py` and `experiments/_nvs_bridge.py`:
this repository's first experiment connecting a *real* neighbour (RFC-MSR01
§8's "No real CLE, HEKB, or NVS-Kernel is connected" gap, closed for
NVS-Kernel). NVS-Kernel is used strictly as a read-only decision oracle
(field/geometry math only, not its risk-tiered `ControlEngine`); this
experiment applies every correction and owns every recorded state itself.
Neither `msr` nor `nvs-kernel` source changed. Field-adapter parity between
the two repos' independent Gaussian-field implementations verified to
`1e-9`; all four fixed perturbation trials recover, re-stabilize (MSR's own
`StabilizationDetector`), converge to within `1e-5` of origin, and land
Lyapunov-contracting; zero quarantines. See `docs/RFC_ALIGNMENT.md` for a
design mistake made and corrected during this work (conflating
transient basin-re-entry with settled-dwell convergence), and for the new
nvs-kernel install step this one experiment — not the core package — now
requires.

### Verified

- TEST_COUNT_PLACEHOLDER `pytest` tests, 100% line and branch coverage,
  mypy strict clean, ruff clean — `src`/`tests` only.
- `experiments/exp_msr_004_recovery.py`: all five EXP-Ubuntu004 metrics
  pass (`"pass": true` in `experiments/results/exp_msr_004.json`).
- Every RFC-MSR01 1.0.0 test still passes unmodified; no existing behavior
  changed.

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
