# ADR 017 spikes (not for merge)

Run with the dev env (DOLFINx 0.10), from the repository root:

    pixi run -e dev python spikes/adr-017/spike_terms.py ../vcell-core/conformance/mechanics/terms.jsonl
    pixi run -e dev python spikes/adr-017/spike_junction.py
    pixi run -e dev python spikes/adr-017/spike_junction_drag.py

- `spike_terms.py`: the FEM shape-derivative forms of decision 1 on P1 meshes whose skeleton is each vcell-core
  complex, against vcell-core's vertex vectors (ADR 017, "A spike"). Result at b0132ea / vcell-core 100ce8c:
  every energy and force to 2.2e-16.
- `spike_junction.py`: an open tense segment ending inside a viscous, drag-screened bulk; the endpoint velocity
  grows by 0.0735 per halving of h (the 2D Kelvin coefficient), ADR 017 decision 3.
- `spike_junction_drag.py`: the same with a point drag mu = 1 at the endpoint; the velocity creeps toward F/mu
  only logarithmically.
