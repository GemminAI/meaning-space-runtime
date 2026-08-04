# RFC Alignment

This document records how this repository's specification (`RFC-MSR01.md`,
at the repo root) relates to the canonical `RFC-MSR00`–`06` series in
`RFCv3_draft/rfc/MSR/`, how each finding from the 2026-08-04 architecture
audit of `src/msr` against that series was disposed of by the **MSR
Strategic Realignment Plan**, and — in the final section — what EXP-Ubuntu004
needed that RFC-MSR01 does not specify. It is a record of decisions already
made, not a design proposal — see `RFC-MSR01.md` for what `src/msr` actually
does.

## Doc identity

**`RFC-MSR01.md` (this repository's root) is authoritative for what
`src/msr` does.** It is written from the implementation (Implementation →
Experiment → Verification → RFC), cites `tests/` and `docs/VERIFICATION.md`
directly, and is kept current with every behavior-affecting change (see its
own §9 Revision History).

`RFCv3_draft/rfc/MSR/RFC-MSR01.md` shares this document's Document-ID and
(pre-1.1.0) version number but is a separate, independently-authored
document — the series' Layer 2 (Architecture) theoretical overview, per
`RFC_MSR_SERIES_INDEX.md` §2's Theory→Architecture→Implementation taxonomy.
The two disagree on mechanics that matter (see "Two Half-Step vs. Unified
SDE" below); this is recorded, not silently reconciled, in both documents'
headers and in `RFC_MSR_SERIES_INDEX.md` §9 item 8.

## Terminology table

Renames between the canonical series (`RFCv3_draft`) and this repository's
ABI (`src/msr/abi.py`). None of these change meaning; they are recorded so a
reader moving between the two doesn't have to re-derive the mapping.

| Canonical (`RFCv3_draft`) | This repository (`msr.abi`) | Note |
|---|---|---|
| `MeaningMeasurement.covariance` | `MeaningMeasurement.sigma` | same quantity, Σ |
| `MeaningMeasurement.trace_id` | `MeaningMeasurement.provenance` | different shape: a single trace ID vs. a tuple of provenance strings |
| `MeaningState.current_basin_id` | `MeaningState.basin_id` | same quantity |
| `MeaningState.step_count` | `MeaningState.step_index` | same quantity |
| `KernelView.state: MeaningState` (nested) | `KernelView`'s fields flattened directly (`theta`, `speed`, `potential`, `gradient`, `basin_id`, ...) | this repository never nests `MeaningState` inside `KernelView` |
| `StabilizedTrajectory.stability_zone_id` | `StabilizedTrajectory.basin_id` | same quantity |
| — | `StabilizedTrajectory.centroid`, `.covariance`, `.dwell_seconds` | present only in this repository's ABI; richer than the canonical minimum |
| "Information Assimilation" (§3.1, canonical) | `InformationAssimilator.assimilate` | same closed-form Bayesian precision fusion |
| "Langevin Flow" | `LangevinFlow.step` | same Euler–Maruyama integrator, restricted to the field prior Φ only (see below) |
| "Semantic Projection" (Meaning Mapper's role, not MSR's) | not present in `msr` | MSR never performs this; `msr.adapters.mapper` only *parses* an already-projected payload |

## Two Half-Step vs. Unified SDE

`RFC-MSR00` §3.2 and `RFC-MSR03` §3.1 specify one Langevin SDE where the
observation and policy terms are inside the same gradient:
`∇V = ∇V_obs + μ∇V_policy`, applied together with the field prior Φ in a
single integration step. `RFC-MSR01` (canonical) §3 and this repository's
implementation instead use two physically separate half-steps:

1. **Information Assimilation** — closed-form, precision-weighted Bayesian
   fusion (`P' = P + Σ⁻¹`, `θ' = P'⁻¹(Pθ + Σ⁻¹θ_obs)`), not a gradient step.
2. **Langevin Flow** — Euler–Maruyama integration on the field prior Φ
   alone (`θ ← θ − γ∇Φ(θ)dt + √(2Tdt)ξ`). No `V_obs` or `V_policy` term
   participates in this step.

**Disposition (Strategic Realignment Plan, 2026-08-04):** the two-half-step
architecture is the implementation's model and is preserved as-is through
Phase 1 — a closed-form Bayesian fusion is the mathematically optimal update
for a linear-Gaussian observation model, which is what `src/msr` assumes.
Whether the unified-SDE formulation should replace it, or whether the RFCs
should instead be revised to describe the two-half-step model, is explicitly
deferred to **Phase 2** ("theoretical standardization") and is not decided
by this document.

## Fisher-Rao Metric vs. Flat Metric

`RFC-MSR00` §2.1, `RFC-MSR01` (canonical) §3.2, and `RFC-MSR03` §3 all
specify natural-gradient descent via the inverse Fisher Information Metric
$g^{-1}(\theta)$. `LangevinFlow` uses a scalar `mobility` (i.e. $g = \gamma^{-1}I$,
a flat Euclidean metric) — there is no generative model $p(x;\theta)$
anywhere in `src/msr` to differentiate for $g_{ij}$.

**Disposition:** deferred to **Phase 2**. This is a genuine open question,
not a known-wrong implementation choice — whether a Fisher-Rao natural
gradient meaningfully improves convergence or stability over the current
flat metric on MSR's actual Gaussian-well fields is an empirical question,
to be settled by a comparison experiment (Fisher vs. flat), not by editing
either the code or the RFCs first.

## Policy Boundary

See `docs/BOUNDARIES.md` for the full disposition — summary: Policy Boundary
enforcement ($V_{\text{policy}}$, the infinite policy wall) is reassigned to
NVS-Kernel, not MSR. No policy mechanism was added to `src/msr` in Phase 1,
and none should be — this is a boundary decision, not an implementation gap.

## Audit finding disposition

Findings from the 2026-08-04 architecture audit (Priority A/B), and what
happened to each:

| Finding | Disposition |
|---|---|
| A-1 — `RFC-MSR01.md` doc-identity collision | **Resolved, Phase 0.** This repository's copy is authoritative for `src/msr`; canonical copy annotated, `RFC_MSR_SERIES_INDEX.md` §9 item 8 records it. |
| A-2 — CFL guard silently clamps instead of failing closed | **Resolved, Phase 1.** `CFLViolation` now raised; see `RFC-MSR01.md` §3.1 and §9. |
| A-3 — Policy Boundary ($V_{\text{policy}}$) unimplemented | **Re-categorized as a boundary decision, Phase 0/1.** Not MSR's responsibility — see `docs/BOUNDARIES.md`. Not implemented in MSR, and Phase 1 explicitly forbids adding it here. |
| A-4 — Fisher-Rao metric vs. flat metric | **Deferred to Phase 2.** Requires an experimental comparison before either the code or the RFCs change. |
| B-1 — `KernelView.gradient` missing | **Resolved, Phase 1.** Field added; see `RFC-MSR01.md` §2, §9. |
| B-2 — `RFC-MSR01`/`RFC-MSR04` give two different formulas for Φ | **Open, Phase 2/3.** RFC-internal; not an implementation defect. Implementation follows `RFC-MSR01`'s formula (verified exact match). |
| B-3 — Unified SDE (`RFC-MSR00`/`03`) vs. two-half-step (`RFC-MSR01`/impl) | **Deferred to Phase 2** — see "Two Half-Step vs. Unified SDE" above. Two-half-step preserved through Phase 1 per explicit instruction. |
| B-4 — Observation map is implicitly $h(\theta)=I$ | **Open, not scheduled.** Reasonable given current Meaning Mapper payloads (full-state, same-space); revisit if that assumption changes upstream. |
| B-5 — Bootstrap cold-start diverges from `RFC-MSR04` §3.1 | **Open, not scheduled.** The implementation's choice (start at the measurement, not a prior well) is deliberate; not revisited in Phase 0/1. |
| B-6 — No property-based testing | **Resolved, Phase 1.** `tests/test_properties.py`, Hypothesis-based, added. |
| B-7 — Internal-state finiteness unchecked past the inbound boundary | **Open, not scheduled.** Boundary-only `isfinite` checks remain; no internal post-step guard was added in Phase 1 (out of the approved scope). |
| C-1 — RFC-MSR02's C-ABI/FFI has no counterpart | **Open, not scheduled.** Scope question for a future binding layer, not `src/msr` itself. |
| C-2 — RFC-MSR05 §3's actuator-domain smoothing algorithm | **Open, not scheduled.** Self-flagged by `RFC-MSR05`'s own header as a likely misplaced concept; no MSR-side action needed. |
| C-3 — Cosmetic ABI field renames | **Resolved, Phase 0.** See terminology table above. |

## EXP-Ubuntu004 additions

`RFC-MSR01.md` is **Verified** and documents the implementation as it stood
at 1.0.0 (2026-08-02): the fast/slow loop, the four ABI types, and the three
verified experiments. Everything in this section was added afterward
(concurrently with the Phase 0+1 work above), for EXP-Ubuntu004 ("OSS Runtime
Validation"), and is **not** part of RFC-MSR01's normative text. Following
this workspace's convention (see `categorical-lift-engine/docs/RFC_ALIGNMENT.md`
for the precedent): a gap between an RFC and its implementation is recorded
here, not silently folded into the RFC or hidden.

### What EXP-Ubuntu004 needed that RFC-MSR01 does not specify

| Addition | Module | Why it's new, not a reinterpretation of RFC-MSR01 |
|---|---|---|
| `FieldPrior.hessian` | `msr.field` | RFC-MSR01 §3.1 only bounds `‖∇²Φ‖` (`stiffness`, for the integrator's step-size guard). The exact `∇²Φ(θ)` at a point was never needed by the verified implementation and is new surface area. |
| `MeaningState.as_dict`/`from_dict`, `StabilizedTrajectory.as_dict`/`from_dict` | `msr.abi` | RFC-MSR01 §2 requires the ABI types to be plain-scalar and serializable "without touching numpy" but never specifies a serialization method — `KernelView.as_dict` was the only precedent. This extends the same pattern to the other two boundary-crossing types. |
| `runtime_stability_rate`, `drift_distance`, `drift_converged` | `msr.metrics` (new) | Neither quantity is mentioned anywhere in RFC-MSR01. Both are EXP-Ubuntu004-specific verification metrics computed *from* the ABI, not new runtime behavior. |
| `largest_lyapunov_exponent` | `msr.lyapunov` (new) | Same: a dynamical-systems Lyapunov exponent is not part of RFC-MSR01's specified behavior. §7/§8 of RFC-MSR01 do not anticipate it. See the module's own docstring for the three approximations this estimate makes (deterministic-drift-only, single-step-Euler-per-recorded-interval, fixed initial perturbation direction) — those are EXP-Ubuntu004-specific measurement-method choices, not claims about the runtime's own behavior. |

### What did not change

`MeaningSpaceRuntime`, `MSRHost`, `InformationAssimilator`, `LangevinFlow`,
`StabilizationDetector`, and the three ports (`msr.ports`) are byte-for-byte
unmodified. Every EXP-Ubuntu004 addition is a pure function or a new method
that *reads* already-computed state (`FieldPrior`, `MeaningState` history) —
none of it participates in `ingest`/`advance`'s step loop, and no existing
test's behavior changed. RFC-MSR01's 95 original 1.0.0 tests still pass
unmodified, now alongside additional tests for both the Phase 0+1 work above
and the EXP-Ubuntu004 surface described here.

### Open question for a future RFC-MSR02 (or an RFC-MSR01 errata)

If EXP-Ubuntu004's metrics prove durable, `msr.metrics` and `msr.lyapunov`
are candidates for RFC-MSR01 §2's ABI or a new §9 — this document does not
claim that promotion, only that the gap exists.

## Phase 3 — NVS-Kernel basin recovery (`experiments/exp_msr_004_recovery.py`)

The first experiment in this repository to connect a *real* neighbour
(closing the "No real CLE, HEKB, or NVS-Kernel is connected" gap RFC-MSR01
§8 states, for NVS-Kernel specifically). `experiments/_nvs_bridge.py`
converts an MSR `FieldPrior` into NVS-Kernel's `WellCache` shape — the two
are structurally the same Gaussian-well model under different field names,
confirmed (not assumed) by a parity check in the experiment itself:
NVS-Kernel's independently-implemented `PotentialField.gradient` is
compared against MSR's own `FieldPrior.gradient` on the same converted
field, at the settled point, before any recovery trial runs. Both `msr` and
`nvs-kernel` remain unmodified; NVS-Kernel is used strictly as a read-only
decision oracle (its field/geometry math), and the experiment applies every
correction itself — see the experiment's own module docstring for why this
uses NVS-Kernel's unconditional field math rather than its risk-tiered
`ControlEngine` (which requires a safe-set/anchor model this synthetic
scenario has no honest way to construct).

**A design mistake made and corrected during this work, recorded rather
than hidden:** the first version of this experiment stopped each recovery
trial as soon as the trajectory re-entered its basin, then reused that same
short, still-transient segment to evaluate Drift Convergence and the
Lyapunov invariant — both of which are asymptotic/settled-state claims. The
result: `drift_converged=False`, `lyapunov_is_contracting=False`, driven
entirely by measuring transient dynamics against thresholds that only make
sense for a genuinely settled dwell. The fix was to let MSR's own
`StabilizationDetector` (already tested, already the ecosystem's definition
of "stabilized") confirm a real quiescent dwell before those three metrics
are evaluated, while still reporting the earlier basin-re-entry step
separately for Basin Recovery Rate (which the spec defines as a bounded
recovery-*time* claim, not a settled-state one). After the fix, all five
EXP-Ubuntu004 metrics pass on all four fixed perturbation trials.

### Gate dependency: nvs-kernel

Running `mypy` or `experiments/exp_msr_004_recovery.py` in this repository
now requires the sibling `nvs-kernel` repository installed into the same
venv (`pip install -e ../nvs-kernel`). This is *not* declared in
`[project.optional-dependencies]` because nvs-kernel is not published to
PyPI and cannot actually be installed that way. `ruff`, `pytest`, and
`mypy` over `src`/`tests` alone (i.e. the 1.0.0 and 1.1.0 surface) do not
need it — only the one new experiment file does. This is a real,
new-as-of-this-experiment environment-setup step, not hidden here.
