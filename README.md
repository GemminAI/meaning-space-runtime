# Meaning Space Runtime (MSR)

Meaning Space Runtime (MSR) is the dynamic Meaning Physics engine of SensOS.
It maintains continuous meaning-state dynamics between Meaning Mapper and
NVS-Kernel while providing stabilized trajectories to CLE and accepting Field
Priors from HEKB.

**MSR does not measure, name, persist or control. It evolves.**

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

> **Implementation → Experiment → Verification → RFC.** This repository
> contains the implementation and experimentally verified runtime. See
> [RFC-MSR01.md](RFC-MSR01.md) for the specification of what is built and
> verified here — it documents this implementation, it does not design it.

## The step

Each step is two physically distinct half-steps:

1. **Assimilation** (information geometry). The measurement is fused in
   information form: `P' = P + Σ⁻¹`, `θ' = P'⁻¹(Pθ + Σ⁻¹θ_obs)`. A sharp
   measurement moves the state; a vague one barely does.
2. **Flow** (meaning physics). The state relaxes on Φ for `dt`:
   `θ ← θ − γ∇Φ(θ)dt + √(2Tdt)ξ`, and positional precision decays toward a
   floor, because information about where the state is ages.

A trajectory that stays quiescent inside one basin for `dwell_steps` is
**stabilized** and handed to CLE. Stabilizing *outside* every basin
(`basin_id is None`) is what seeds new knowledge in Bootstrap Mode.

## Layout

| Module | Role |
|---|---|
| `msr.abi` | The four frozen boundary types |
| `msr.field` | Φ as Gaussian wells, basins, curvature bound |
| `msr.dynamics` | Assimilation + Langevin flow with the stiffness guard |
| `msr.stability` | Stabilization detection (latched, one emission per dwell) |
| `msr.runtime` | `MeaningSpaceRuntime` — one instance owns one frame |
| `msr.ports` | Structural protocols for kernel / CLE / HEKB |
| `msr.host` | `MSRHost` — routes both loops |
| `msr.adapters` | Mapper payload in, HEKB concepts → field prior |
| `msr.reference` | Non-normative in-process stand-ins, for experiments only |

MSR imports no neighbour code. The ports are `Protocol`s, so NVS-Kernel, CLE
and HEKB satisfy them by shape.

## Use

```python
from msr import MeaningMeasurement, MeaningSpaceRuntime, MSRHost

runtime = MeaningSpaceRuntime(frame_id="F", dimension=8)
host = MSRHost(runtime=runtime, kernel=my_kernel, cle=my_cle, hekb=my_hekb)

result = host.ingest(MeaningMeasurement.isotropic("obs-1", "F", theta, 0.1, ts))
result.kernel_view      # went to NVS-Kernel
result.stabilized       # went to CLE, if this step stabilized
```

## Verification

```bash
.venv/bin/python -m pytest tests -q --cov=msr --cov-report=term-missing
```

Current: **95 tests pass, 100% line and branch coverage, mypy strict clean,
ruff clean.**

## Experiments

```bash
.venv/bin/python experiments/exp_msr_003_developmental_loop.py
```

Results land in `experiments/results/*.json`. All three experiments are
deterministic (temperature 0; every noise stream is seeded). See
[docs/VERIFICATION.md](docs/VERIFICATION.md) for measured numbers and the two
defects the experiments exposed, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the module-level architecture as built.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
