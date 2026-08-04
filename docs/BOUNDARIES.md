# Boundaries

What MSR owns, what it deliberately does not, and where each neighbor's
responsibility begins. This is a documentation record of a boundary
decision (Strategic Realignment Plan, Phase 0, 2026-08-04), not a design
proposal — nothing here changes `src/msr`'s code.

## The pipeline

```
MeasuredObservation
      │
      ▼
Meaning Mapper           — semantic projection only
      │
      ▼ MeaningMeasurement
Meaning Space Runtime     — continuous state evolution only
      │              │
      │              ▼ KernelView (every step)
      │         NVS-Kernel — stabilization *control* + Policy Boundary enforcement
      ▼
StabilizedTrajectory (on dwell)
      │
      ▼
CLE                       — knowledge crystallization only
      │
      ▼
HEKB                       — persistent knowledge only
```

## MSR's responsibility: pure phase-space geometric trajectory tracking

MSR's essential responsibility is tracking a continuous trajectory through
meaning space: fusing measurements into a belief state (`InformationAssimilator`)
and relaxing that state along a HEKB-supplied potential field
(`LangevinFlow`), then reporting what it observes (`KernelView`) and what
stabilized (`StabilizedTrajectory`). It does not evaluate, decide, or
constrain — see `RFC-MSR01.md` §1: MSR **MUST NOT** measure, name/persist
knowledge, or issue control decisions.

| Neighbor | Owns | Verified how |
|---|---|---|
| Meaning Mapper | Semantic projection (observation → θ) | MSR never derives θ from anything lower-level; `msr.adapters.mapper.measurement_from_payload` only parses an already-projected payload |
| **MSR** | **Continuous state evolution** (assimilation + flow + stabilization detection) | No import of Meaning Mapper/NVS-Kernel/CLE/HEKB code anywhere in `src/msr`; every neighbor is a `typing.Protocol` in `msr.ports` |
| NVS-Kernel | Stabilization *control* and Policy Boundary enforcement | Reads `KernelView` (now including `gradient`, §9 of `RFC-MSR01.md`); MSR issues no control decisions |
| CLE | Knowledge crystallization (naming, lifting) | MSR never mints a concept ID; `msr.reference.ReferenceCLE` is explicitly a non-normative test/experiment stand-in |
| HEKB | Persistent knowledge | MSR holds no persistent store; `FieldPrior` is pulled fresh via `FieldPriorSource.field_prior(...)` each refresh, never written back to by MSR |

## Policy Boundary belongs to NVS-Kernel, not MSR

`RFC-MSR00` §5 and `RFC-MSR03` §4.3 (canonical series) describe a
constitutional invariant — a policy potential $V_{\text{policy}}(\theta) \to
\infty$ forming an impenetrable wall, enforced by reflecting any state that
approaches it. `src/msr` implements none of this: no policy potential, no
$\mu$ weighting term, no reflective-boundary projection, no
`MSR_ERR_BOUNDARY_PENETRATION`-equivalent.

**This is a boundary decision, not a gap to close.** MSR's role is reporting
geometry — `potential`, `gradient`, `basin_id` — accurately and promptly.
Deciding what to *do* about a trajectory approaching a forbidden region
(reject it, reflect it, halt it, escalate it) is a control decision, and
control decisions are NVS-Kernel's responsibility per `RFC-MSR01.md` §1.
A future NVS-Kernel can read `KernelView.gradient` and `KernelView.potential`
every step and enforce whatever policy it owns without MSR needing to know
what that policy is.

Consequently: **Phase 1 explicitly does not add any Policy Boundary
mechanism to `src/msr`**, and none should be added here in a future phase
either — if `RFC-MSR00`/`RFC-MSR03`'s text is ever reconciled with this
decision, the reconciliation is "cite NVS-Kernel as the owner," not "move
the algorithm into MSR." See `RFC_MSR_SERIES_INDEX.md` §9 item 9 for the
series-wide record of this same decision answering that index's own
previously-open item 6.

## What this document does not change

No code in `src/msr` changed as a result of this document. It exists so the
boundary is written down somewhere both the RFC series and this repository
can point to, instead of being re-derived from scratch by the next reader
(or the next audit).
