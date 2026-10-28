# SkinMRASim

Bloch simulation of time-of-flight contrast for small, slow-flowing vessels in
skin at 3 T. Tissue properties are held as literature intervals rather than
single values, so every result is a distribution over parameter draws.

There is no k-space encoding and no image synthesis. The point spread function
enters only through partial volume, as a stated vessel-to-voxel fraction.

## Install

```bash
pip install numpy scipy matplotlib pytest
python -m pytest
```

Python 3.11 or newer. 100 tests, about a minute.

## Modules

| module | contents |
|---|---|
| `magnetisation.py` | affine maps on longitudinal magnetisation: composition, repetition, fixed points |
| `tissues.py` | tissue properties as intervals with sources, and the Latin hypercube sampler |
| `protocol.py` | the sequence, and how k-space is covered |
| `geometry.py` | path through the slab, dwell time, laminar shells, skin layer depths |
| `model.py` | signal, contrast, partial volume, CNR, efficiency |
| `optimise.py` | grid sweeps and continuous search |
| `plotting.py` | shared figure style |

The stack runs one way. `magnetisation` contains no MR. `tissues`, `protocol`
and `geometry` describe the problem. `model` is where they meet. `optimise`
sits on top and nothing imports it.

`tests/reference.py` holds independent implementations of the same physics,
written from textbook expressions and from a spin marched one excitation at a
time. It imports nothing from `skinsim`, and `skinsim` imports nothing from it.

## Figures

```bash
python figures/inflow.py
python figures/optimisation.py
```

`inflow.py` plots blood signal against mean velocity for a 0.2 mm vessel
crossing an 11.44 mm slab head-on, over 64 velocities from 2 to 300 mm/s at
TR 35.06 ms, TE 5.86 ms and 25°. Four velocity-independent levels are drawn
alongside it: stagnant blood, subcutaneous fat, fat after suppression, and
dermis. The gap between the curve and stagnant blood is the inflow
enhancement; the crossings with the tissue levels are solved by bisection in
log velocity rather than read off the plotting grid.

`optimisation.py` maps contrast-to-noise per square root of scan time over a
36 x 36 grid of flip angle (3 to 45°) and repetition time (12 to 260 ms), at
blood velocities of 2, 8 and 30 mm/s, for 0.25 mm in-plane resolution, a 50 mm
phase field of view, twofold acceleration and g = 1.3.

Both scripts sample the tissue intervals by Latin hypercube with seed
20260422, take the median over draws, and record the seed and draw count on
the image. Every parameter is read from `tissues.py`, `protocol.py` and
`geometry.py`, never from a literal in the plotting code, so a figure cannot
drift from the model.

## Licence

MIT. See [LICENSE](LICENSE).
