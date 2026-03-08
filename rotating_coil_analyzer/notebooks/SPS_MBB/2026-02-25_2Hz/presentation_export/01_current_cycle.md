# Current Cycle with Turn Classification

The SPS supercycle ramps the magnet current from ~300 A (injection) to ~4800 A
(flat-top) and back. The rotating coil measures continuously at 2 Hz.

Only turns sitting entirely on a **current plateau** (dI/dt ≈ 0) are
usable for precision field measurement. During ramps the field changes within
one rotation, mixing eddy currents with the quasi-static field.

- **Green:** injection plateau (~300 A) — used for eddy settling + low-field harmonics
- **Blue:** flat-top plateau (~4800 A) — used for high-field harmonics
- **Red:** ramp turns — rejected

![Current cycle](current_cycle.png)
