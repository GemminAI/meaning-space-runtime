# MSR Verification Record

Every number below was produced by running the stated command in this
repository. Nothing is estimated.

Environment: macOS 25.5.0 (darwin), Python 3.12.12, numpy 2.x, `.venv`.

## Static and unit verification

| Check | Command | Result |
|---|---|---|
| Test suite | `python -m pytest tests -q` | **95 passed** |
| Coverage | `python -m pytest tests --cov=msr` | **100%** lines (727/727), **100%** branches (156/156) |
| Types | `python -m mypy` (strict, src+tests+experiments) | **Success**, 27 files |
| Lint | `python -m ruff check .` | **All checks passed** |
| Format | `python -m ruff format --check .` | 28 files already formatted |

## Experiments

All three are deterministic: flow temperature is 0 and every noise stream is
seeded, so reruns reproduce these numbers exactly.

### EXP-MSR-001 — fast loop, field convergence · **PASS**

A trajectory released 1.98 units from a known well, 200 field-only steps:

| Metric | Value |
|---|---|
| Distance to well mean | 1.98 → 7.5e-19 |
| Potential Φ | −0.563 → −4.000 (well depth) |
| Φ monotonically decreasing | **true** (all 201 steps) |
| Basin reported to kernel | `c-known` |

The field attracts, the potential never increases, and the basin MSR reports to
NVS-Kernel agrees with the geometry.

### EXP-MSR-002 — bootstrap and its control arm · **PASS**

| Arm | Stabilized | Lifts | Concepts | Final speed |
|---|---|---|---|---|
| Stationary measurement, empty HEKB | step 4 | 2 (steps 4, 9) | `concept-001` | **0.0** |
| Drifting measurement (control) | **never** | 0 | none | 1.098 |

With Φ ≡ 0 the runtime still stabilizes and the trajectory is marked novel
(`basin_id is None`) — that novelty is what gives CLE something to lift. The
lift at step 4 seeds the concept; the lift at step 9 reinforces it once the
field arrives; then the loop reaches a fixed point (0 lifts in the final 10
steps). The drifting control arm never stabilizes, so the detector is measuring
something real.

### EXP-MSR-003 — slow loop, developmental effect · **PASS**

Identical, byte-for-byte seeded noisy measurement stream (σ = 0.05) played to
two arms that differ only in whether phase 1 happened:

| Arm | Knowledge before phase 3 | Stabilized in phase 3 | Settled speed | Final basin |
|---|---|---|---|---|
| Experienced | concept at A = (1.0, −0.5) | **step 28** | **0.000** | `concept-001` |
| Naive | concept at B = (−1.5, 1.0) | **never** | 0.107 | none |

Prior exposure to A makes a later, noisier observation of A resolvable: the
concept's well absorbs the measurement jitter that flat space cannot. This is
the RFC-SensOS24 developmental claim, measured.

## Defects found by the experiments and fixed

Both were found by running EXP-MSR-002, not by inspection. Both have dedicated
regression tests in `tests/test_stiffness_and_latch.py`.

### 1. Stiff-field divergence

A concept lifted from a near-stationary dwell has a near-singular covariance,
whose inverse is an almost-delta well (precision ~1e6 at the original ridge).
Explicit Euler on such a well overshoots by ~19× per step, and the state
oscillated instead of settling — a legitimately-committed concept blew up the
runtime that committed it. Observed as a residual speed of 0.316 where 0 was
expected.

Fixed at two layers:

- `LangevinFlow` now subdivides each step so that `γ·stiffness·sub_dt ≤
  stability_factor` — a CFL-style condition on the field's **curvature**
  (`FieldPrior.stiffness`, a `Σ dᵢλ_max(Pᵢ)` bound computed once per field),
  plus a displacement bound for the far field. Total noise variance is
  preserved across substeps (verified to within 5% over 20 000 samples).
- The reference CLE floors a lifted concept's width at `MIN_CONCEPT_WIDTH`:
  knowledge must not be sharper than the measurements that produced it.

An intermediate displacement-only guard was tried first and did **not** fix it
— displacement is the wrong criterion for explicit-integrator stability. The
first fix was measured, found insufficient, and replaced.

### 2. Slow-loop runaway

`set_field_prior` originally dropped the stabilization latch unconditionally.
That made the slow loop self-triggering: CLE reinforces a concept → HEKB's
version bumps → MSR drops the latch → the same motionless state re-stabilizes →
CLE reinforces again. EXP-MSR-002 showed 6 lifts and climbing from a runtime
that never moved.

`set_field_prior` now drops the latch **only when the new field puts the current
position in a different basin**, and returns whether it did. A field change that
does not reclassify the state leaves the dwell a valid statement. Post-fix the
loop settles at exactly 2 lifts (seed, reinforce) and then stops.

## Not verified

- **No real neighbour was connected.** CLE, HEKB and NVS-Kernel are represented
  by `msr.reference` stand-ins. The ports are structural, and the HEKB adapter
  reads the `Concept` shape (`id`/`centroid`/`hessian`/`invariants`) that
  exp7100 already emits, but no cross-process or cross-repo integration was run.
- **No performance measurement.** No allocation-free or fixed-rate claim is made
  here; the substepping guard makes per-step cost field-dependent by design.
- **Dimension**: experiments run at d = 1 and d = 2. Nothing in the code is
  dimension-specific, but higher-dimensional behaviour is untested.
- **Frame origin**: MSR requires a declared frame and never rebases. What
  defines the frame in Bootstrap Mode remains an open question upstream, in
  Meaning Mapper, not here.
