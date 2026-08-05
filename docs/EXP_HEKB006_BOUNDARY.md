# EXP-HEKB006 — Repository Boundary Report

**Status: NOT APPLICABLE to this repository.**

This document exists so that a reader encountering EXP-HEKB006 v2.1.0
("Observer Invariance, Epistemic Agnosticism & Reality-Driven Multi-Observer
Convergence Validation") in connection with `meaning-space-runtime` finds a
recorded disposition instead of a partially-built substitute implementation.
Following this workspace's convention (see `docs/BOUNDARIES.md` and
`docs/RFC_ALIGNMENT.md`), a boundary decision is written down, not silently
worked around.

## What EXP-HEKB006 requires

The specification's pipeline is:

```
Reality Object Q
      │
      ▼
Observer M_k (Human panel, Gemma, Qwen, Llama, Mistral, Claude, GPT, Gemini)
      │
      ▼
Meaning Mapper
      │
      ▼
MSR Trajectory
      │
      ▼
Categorical Lift Engine (CLE)
      │
      ▼
Reality Consensus Engine (Pullback Limit / Intersection of Typed Morphisms)
      │
      ▼
Observer Invariance Verification  S(Q)^(Human) ≡ S(Q)^(Claude) ≡ … ≡ S(Q)
```

It further assumes, as already-completed prior work to be reused:

- Real observed-reality corpora (masterpiece visual art, a multi-modal music
  corpus, software architecture/spec artifacts) from EXP-HEKB003/004/005.
- A Human Expert Ground Truth panel's extracted semantic closures.
- A working `meaning-mapper`, `cle-core` (Categorical Lift Engine),
  Semantic Closure Engine, Semantic Search Engine, HEKB Storage Engine, and
  HEKB MCP Service, all already integrated with each other.

## What this repository actually contains

`meaning-space-runtime` implements exactly one stage of that pipeline: **MSR**
— continuous phase-space trajectory tracking (`src/msr`). Per
`docs/BOUNDARIES.md`, MSR's stated responsibility is narrow by design:

| Neighbor | Owns | Present in this repo? |
|---|---|---|
| Meaning Mapper | Semantic projection (observation → θ) | No — `msr.adapters.mapper` only *parses* an already-projected payload; it does not perform projection. |
| **MSR** | **Continuous state evolution** | Yes — this is the whole repo. |
| NVS-Kernel | Stabilization control | No (external sibling repo, used read-only in `experiments/exp_msr_004_recovery.py`). |
| CLE | Knowledge crystallization (naming, lifting) | No — `msr.reference.ReferenceCLE` is explicitly documented as a non-normative test stand-in, not a real CLE. |
| HEKB | Persistent knowledge storage | No — `src/msr/adapters/hekb.py` is a 117-line structural adapter (`Concept` → `GaussianWell`) built against a `typing.Protocol`. Its own docstring states HEKB is never imported. There is no storage engine, no closure engine, no search engine, no MCP service behind it. |

Additionally:

- No corpora (visual art, music, or software-spec) exist in this repository.
- No experiments named `EXP-HEKB001`–`EXP-HEKB005` exist here — the only
  completed experiments are `exp_msr_001`–`004` (field convergence,
  bootstrap, developmental loop, NVS-Kernel basin recovery), which are MSR
  dynamics validations unrelated to observer/LLM comparison.
- No human-expert ground-truth data, no multi-LLM observer adapters, and no
  Reality Consensus Engine exist here.

The separate RFC series at `RFC-HEKB00`–`14` (+ series index) that motivates
this experiment is not reflected in this repository's code or git history
either; it documents a different, independently-maintained system (the HEKB
repository).

## Disposition

EXP-HEKB006 is a repository-level validation experiment belonging to the
**HEKB repository**, where `meaning-mapper`, `cle-core`, the Semantic
Closure Engine, and EXP-HEKB001–005's completed corpora and results actually
exist. `meaning-space-runtime` is one upstream producer feeding into that
system (via the boundary in `docs/BOUNDARIES.md`), not a place to
reimplement it.

Consistent with that:

- **No** substitute `meaning-mapper`, `cle-core`, Semantic Closure Engine,
  Reality Consensus Engine, or observer adapters have been added to this
  repository.
- **No** synthetic corpora, fabricated human-expert data, or placeholder
  metrics have been generated to imitate EXP-HEKB001–005's outputs.
- `src/msr/adapters/hekb.py` is unchanged — it remains a structural,
  protocol-only adapter, as designed.

**Marked: NOT APPLICABLE for `meaning-space-runtime`.** Execution of
EXP-HEKB006 should proceed inside the HEKB repository, where its
prerequisite phases (EXP-HEKB001–005) already exist. If a future phase of
EXP-HEKB006 needs something specific from MSR — e.g., a trajectory-level
invariant beyond what `KernelView`/`StabilizedTrajectory` already expose —
that should be raised as a request against MSR's existing ABI (`RFC-MSR01.md`,
`docs/BOUNDARIES.md`), not built as a parallel implementation in either
repository.
