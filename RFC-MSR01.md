# RFC-MSR01: Meaning Space Runtime — Implementation Specification

| Field | Value |
|---|---|
| RFC ID | RFC-MSR01 |
| Version | 1.1.0 |
| Status | Verified — documents a validated implementation |
| Topic | Meaning Space Runtime (MSR): fast-loop state evolution and slow-loop knowledge closure |
| Repository | [`GemminAI/meaning-space-runtime`](https://github.com/GemminAI/meaning-space-runtime) |
| Neighbors | Meaning Mapper (upstream, L1→L2), NVS-Kernel (fast-loop downstream), CLE (slow-loop downstream), HEKB (slow-loop closure, upstream via Field Prior) |

**Process note.** This document follows the Implementation → Experiment →
Verification → RFC sequence: every claim below is backed by a passing test in
`tests/` or a measured result in `docs/VERIFICATION.md`, not by design intent.
It specifies what this repository already does, not what a future version
should do.

**Normative keywords** follow RFC 2119 (`MUST` / `SHOULD` / `MAY`).

**Doc-identity note.** The `RFCv3_draft` workspace's `rfc/MSR/RFC-MSR01.md`
shares this document's ID and (pre-1.1.0) version number but is a separate,
independently-authored document with materially different content — see
`docs/RFC_ALIGNMENT.md` for the reconciliation. **This document is
authoritative for what `src/msr` actually does**; the `RFCv3_draft` copy is
the series' theoretical overview, reconciled with this one only as far as
`docs/RFC_ALIGNMENT.md` records.

## 1. Purpose

MSR maintains continuous meaning-state dynamics between Meaning Mapper and
NVS-Kernel, and closes the developmental loop between meaning-state dynamics
and HEKB via CLE:

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

MSR **MUST NOT** measure (Meaning Mapper's role), name or persist knowledge
(CLE's / HEKB's role), or issue control decisions (NVS-Kernel's role). MSR
**MUST NOT** import code from any of the three neighbors; every crossing
**MUST** go through the ABI (§2) via a structural port (§4).

## 2. ABI

Four frozen types, defined in `msr.abi`, cross MSR's boundaries. All fields
are plain Python scalars/tuples — no numpy types leak across the boundary.

| Type | Direction | Carries |
|---|---|---|
| `MeaningMeasurement` | Meaning Mapper → MSR | `observation_id`, `frame_id`, `theta` (θ), `sigma` (Σ, SPD), `timestamp_ns`, `provenance` |
| `MeaningState` | Internal / observable | `frame_id`, `step_index`, `time_s`, `theta`, `precision`, `speed`, `potential`, `basin_id`, `source_observation_id` |
| `KernelView` | MSR → NVS-Kernel, every step | `frame_id`, `step_index`, `time_s`, `theta`, `speed`, `potential`, `gradient`, `basin_id`, `precision_trace`, `stabilized` |
| `StabilizedTrajectory` | MSR → CLE, on stabilization | `trajectory_id`, `frame_id`, `basin_id`, `states`, `centroid`, `covariance`, `dwell_steps`, `dwell_seconds`, `provenance` |

A `MeaningMeasurement` **MUST** carry a declared `frame_id` and a symmetric
positive-definite `sigma`; `MeaningMeasurement.validate()` enforces this and
**MUST** be called (the runtime calls it on every `ingest`). A measurement in
a frame other than the runtime's own frame **MUST** be rejected
(`FrameMismatch`).

`StabilizedTrajectory.basin_id is None` denotes a **novel** stabilization —
one that occurred outside every basin the current field prior defines. This
is verified behavior: EXP-MSR-002 (§6) shows an empty field prior (Bootstrap
Mode) still produces a novel stabilization, giving CLE something to lift with
no prior knowledge at all.

## 3. The step

Each call to `MeaningSpaceRuntime.ingest` performs, in order:

1. **Assimilation.** The measurement is fused into `(θ, P)` in information
   form: `P' = P + Σ⁻¹`, `θ' = P'⁻¹(Pθ + Σ⁻¹θ_obs)`.
2. **Decay.** Positional precision relaxes toward `precision_floor` with time
   constant `forgetting_tau`.
3. **Flow.** `θ` relaxes on the field prior Φ for the elapsed `dt`:
   `θ ← θ − γ∇Φ(θ)dt + √(2Tdt)ξ`, subject to the stiffness guard (§3.1).
4. **Reporting.** A `MeaningState` is computed and appended to history; a
   `KernelView` **MUST** be emitted for every step; a `StabilizedTrajectory`
   **MUST** be emitted on, and only on, the step that latches a new dwell
   (§3.2).

`MeaningSpaceRuntime.advance` performs steps 2–4 without a measurement, for a
kernel that ticks faster than the mapper measures.

### 3.1 Stiffness guard

The field prior Φ is supplied by HEKB and is not bounded by MSR at the point
of authorship — a committed concept may be arbitrarily sharp. An explicit
integration step therefore **MUST** be subdivided so that, for each substep,
`γ · stiffness(Φ) · sub_dt ≤ stability_factor`, where `stiffness(Φ) = Σᵢ dᵢ ·
λ_max(Pᵢ)` bounds `‖∇²Φ‖` over the whole space, plus a displacement bound for
the far field. Total injected noise variance **MUST** be preserved regardless
of substep count. Without this guard, a legitimately committed concept can
make the runtime that committed it diverge (§7, defect 1).

The guard **MUST** fail closed: if the substep count a field's curvature
requires exceeds `max_substeps`, `LangevinFlow.substep_count` (and therefore
`.step`, and therefore `ingest`/`advance`) **MUST** raise `CFLViolation`
rather than silently truncating to `max_substeps`. A truncated count no
longer satisfies the stability condition it was derived from, so continuing
would integrate an unstable step under the appearance of a stable one. Prior
to v1.1.0 this guard silently clamped instead of raising — see §9, Priority
A‑2 of the architecture audit this revision resolves.

### 3.2 Stabilization

A trajectory is **stabilized** when it has remained within
`speed_threshold` and inside one basin for `dwell_steps` consecutive states.
The detector **MUST** latch: exactly one `StabilizedTrajectory` is emitted
per dwell, and no further emission occurs until the dwell is broken (by
motion or a basin change) and re-forms.

Installing a new field prior (`set_field_prior`) **MUST NOT** unconditionally
reset this latch. It **MUST** reset the latch only when the new field
reclassifies the current position into a different basin (or before any
measurement has been ingested). A field update that leaves the current basin
unchanged **MUST** leave the dwell valid. Resetting unconditionally makes the
slow loop self-triggering (§7, defect 2).

## 4. Ports

Three `typing.Protocol` types (`msr.ports`) are the only contact MSR has with
its neighbors:

- `KernelPort.observe(view: KernelView) -> None` — fast loop.
- `LiftPort.lift(trajectory: StabilizedTrajectory) -> None` — slow loop, outbound.
- `FieldPriorSource.field_prior(frame_id, dimension) -> FieldPrior | None` — slow loop, inbound.

Any object satisfying a port's shape **MAY** be bound to `MSRHost`; MSR
**MUST** function with any subset of ports absent (headless / knowledge-frozen
/ lift-free operation is a tested configuration, not a degraded one).

## 5. Adapters

- `msr.adapters.mapper.measurement_from_payload` **MUST** accept both native
  field names (`theta`/`sigma`) and annotation-era names still emitted by
  deployed mappers (`coordinates`/`covariance`/scalar `variance`), and
  **MUST** reject a payload missing a declared frame, position, or
  uncertainty — nothing is inferred.
- `msr.adapters.hekb.well_from_concept` **MUST** build an anisotropic well
  from a concept's Hessian when it is valid SPD, and **MUST** fall back to an
  isotropic well otherwise (including when the Hessian is present but not
  SPD — degraded, not trusted). A concept with no centroid **MUST** be
  skipped, not guessed at.

## 6. Verified behavior

Full detail and measured numbers: [docs/VERIFICATION.md](docs/VERIFICATION.md).
Summary:

| Experiment | Claim | Result |
|---|---|---|
| EXP-MSR-001 | A known field attracts a trajectory; potential decreases monotonically; reported basin agrees with geometry | **PASS** |
| EXP-MSR-002 | An empty field (Bootstrap Mode) still stabilizes and marks the result novel; a drifting control never stabilizes | **PASS** |
| EXP-MSR-003 | Prior exposure to a concept makes a later, noisier observation of it resolvable, vs. a naive arm on the byte-identical stream | **PASS** |

Static verification: 95 tests, 100% line and branch coverage, mypy strict
clean, ruff clean.

## 7. Defects found and fixed during verification

1. **Stiff-field divergence.** A concept lifted from a near-stationary dwell
   produced a near-delta well; explicit Euler overshot ~19× per step. Fixed
   by the stiffness guard (§3.1) and a minimum-width floor on lifted
   concepts. A displacement-only guard was tried first, measured, and found
   insufficient before this fix was adopted.
2. **Slow-loop runaway.** An unconditional latch reset on every field update
   made reinforcement self-triggering (unbounded duplicate lifts from a
   motionless state). Fixed by the basin-change-gated reset policy (§3.2).

Both have dedicated regression tests in `tests/test_stiffness_and_latch.py`.

## 8. Not covered by this specification

- No real CLE, HEKB, or NVS-Kernel is connected; `msr.reference` provides
  non-normative in-process stand-ins used only by tests and experiments.
- No performance/allocation claim is made; the stiffness guard makes
  per-step cost field-dependent by design.
- Only `dimension ∈ {1, 2}` has been exercised experimentally.
- What defines a frame in Bootstrap Mode (before any measurement) is an open
  question upstream, in Meaning Mapper, not resolved here.

## 9. Revision History

### 1.1.0 (2026-08-04) — Strategic Realignment Plan, Phase 0 + Phase 1

Implements the Phase 0 (documentation) and Phase 1 (core reliability) items
of the MSR Strategic Realignment Plan, itself a response to an architecture
audit of `src/msr` against `RFC-MSR00`–`06` (`RFCv3_draft/rfc/MSR/`). See
`docs/RFC_ALIGNMENT.md` for the full disposition of every audit finding.

- **Breaking.** The stiffness guard (§3.1) now fails closed:
  `CFLViolation` is raised instead of silently clamping to `max_substeps`.
- **Breaking.** `KernelView` (§2) gained a required `gradient` field —
  ∇Φ(θ) at the reported position — so NVS-Kernel does not have to
  recompute the field's pull direction against its own copy of Φ.
- Added a doc-identity note (above) resolving the collision with
  `RFCv3_draft/rfc/MSR/RFC-MSR01.md`: this document is authoritative for
  `src/msr`'s verified behavior.
- 106 tests (95 → 106; new: property-based tests via Hypothesis covering
  determinism/replay, SPD/information-fusion invariants, the analytic
  gradient against finite differences, and CFL bound satisfaction / fail-
  closed behavior), 100% line and branch coverage maintained, mypy strict
  clean, ruff clean.
- Explicitly **not** changed: the two-half-step architecture (§3) is
  preserved as-is; no Policy Boundary ($V_{\text{policy}}$) mechanism was
  added to MSR (see `docs/BOUNDARIES.md` — that responsibility is assigned
  to NVS-Kernel, not MSR); no Fisher-Rao metric was added (deferred to
  Phase 2, pending an experimental comparison against the current flat
  metric). These are open items, not oversights — see
  `docs/RFC_ALIGNMENT.md`.

### 1.0.0 (2026-08-02)

Initial specification of the validated implementation (§1–§8 as originally
published).
