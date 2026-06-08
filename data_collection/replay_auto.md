# Parametric Human-Like Trajectory Synthesis for Contact Quality Assessment

## 1. Motivation

In contact-rich manipulation, the quality of collected demonstrations directly impacts policy learning performance. Human teleoperation produces natural but inconsistent data, while naive programmatic trajectories (linear interpolation, constant velocity) create an undesirable domain gap with real-world execution. 

We propose a **parametric trajectory synthesis framework** that bridges this gap: generating trajectories with human-like kinematic properties (smooth curvature, compliance, micro-variation) while maintaining precise control over contact quality labels (positive/negative).

---

## 2. Problem Formulation

We define an end-effector trajectory as a time-indexed sequence:

$$\tau = \{p_t\}_{t=0}^{T}, \quad p_t = (x, y, z, r_x, r_y, r_z) \in \mathbb{R}^6$$

where $(x,y,z)$ denotes Cartesian position (meters) and $(r_x, r_y, r_z)$ denotes orientation in Euler angles (radians). The control frequency is $f_c = 20$ Hz.

**Objective**: Given a task-specific parameter set $\Theta = \{\theta_i \pm \Delta\theta_i\}$, synthesize trajectory families $\{\tau^{(k)}\}_{k=1}^{N}$ such that:
1. Each $\tau^{(k)}$ is kinematically smooth (bounded jerk)
2. Inter-trajectory variation matches human demonstration statistics
3. Contact quality labels are precisely controlled

---

## 3. Trajectory Synthesis via Cubic Bézier Primitives

### 3.1 Cubic Bézier Curve Formulation

All non-contact motion segments (approach, return, pass-transition) are parameterized as cubic Bézier curves. Given start point $P_0$ and end point $P_3$, the trajectory is:

$$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t \cdot P_1 + 3(1-t)t^2 \cdot P_2 + t^3 P_3, \quad t \in [0,1]$$

where $P_1, P_2$ are control points that determine the curvature. This guarantees:
- $C^1$ continuity at endpoints (tangent continuity)
- Smooth acceleration profile (no instantaneous velocity jumps)
- Bounded curvature (physically realizable by a robot arm)

### 3.2 Stochastic Control Point Placement

To produce human-like variability, control points are sampled from a structured distribution:

$$P_1 = P_0 + \alpha(P_3 - P_0) + \epsilon_1, \quad \alpha = 0.3$$
$$P_2 = P_0 + \beta(P_3 - P_0) + \epsilon_2, \quad \beta = 0.7$$

where $\epsilon_i \sim \mathcal{U}([-\delta_x, \delta_x] \times [-\delta_y, \delta_y] \times [-\delta_z, \delta_z])$ with task-specific bounds:

| Segment | $\delta_x$ (mm) | $\delta_y$ (mm) | $\delta_z$ (mm) |
|---------|---------|---------|---------|
| Approach | [-5, 10] | [-5, 5] | [-10, 5] |
| Return | [-5, 10] | [-5, 5] | [0, 10] |
| Pass-transition | $\pm x_{overshoot}$ | — | [1, 2] (lift) |

The asymmetric bounds encode biomechanical priors: approach tends to arc forward-and-down, return tends to lift rapidly.

### 3.3 Temporal Discretization

The curve parameter $t$ is uniformly sampled at $N$ points:

$$N = \max\left(\left\lfloor \frac{\|P_3 - P_0\|}{v_{seg}} \right\rfloor, \, N_{min}\right)$$

where $v_{seg}$ is the segment-specific velocity (m/step) and $N_{min} = 40$ ensures sufficient smoothness even for short segments.

---

## 4. Contact-Phase Dynamics Modeling

### 4.1 Parametric Randomization

Each trajectory instance samples its kinematic parameters independently:

$$\theta_i^{(k)} = \bar{\theta}_i + \mathcal{U}(-\Delta\theta_i, +\Delta\theta_i)$$

| Parameter $\theta_i$ | Base $\bar{\theta}_i$ | Jitter $\Delta\theta_i$ | Physical meaning |
|---|---|---|---|
| $z_c$ (contact height) | 125 mm | ±1 mm | Surface compliance variation |
| $x_0$ (wipe start) | 270 mm | ±5 mm | Spatial variability |
| $x_1$ (wipe end) | 420 mm | ±5 mm | Spatial variability |
| $y_0$ (lateral start) | -10 mm | ±1 mm | Lateral offset |
| $\Delta y$ (pass gap) | 8 mm | ±1 mm | Inter-pass spacing |
| $v_w$ (wipe velocity) | 10 mm/s | ±10% | Speed variability |
| $v_a$ (approach velocity) | 16 mm/s | ±10% | Speed variability |

These ranges are derived from statistical analysis of 80 human-collected episodes (dataset `260522_v8l_caheiban`), ensuring the synthetic distribution envelops the real one.

### 4.2 Contact Compliance Model (Z-axis)

During surface contact, the Z-coordinate exhibits low-frequency oscillation arising from surface micro-geometry and wrist compliance. We model this as a weighted dual-sinusoid:

$$z(t) = z_c + A_z \left[ w_1 \sin(2\pi f_1 t + \phi_1) + w_2 \sin(2\pi f_2 t + \phi_2) \right]$$

where:
- $A_z = 1.5$ mm (compliance amplitude, matching real data: $\sigma_z = 0.27$ mm, peak-to-peak $\approx 1.4$ mm)
- $f_1 \sim \mathcal{U}(0.05, 0.10)$ Hz — primary mode (period 10–20 s)
- $f_2 \sim \mathcal{U}(0.15, 0.25)$ Hz — secondary mode (period 4–7 s)
- $w_1 = 0.7, \, w_2 = 0.3$ — weighting (dominant slow mode)
- $\phi_1, \phi_2 \sim \mathcal{U}(0, 2\pi)$ — random phase offsets

**Design rationale**: Spectral analysis of real contact trajectories reveals energy concentration below 0.3 Hz. Higher frequencies ($>1$ Hz) manifest as mechanical vibration, not human-like compliance.

### 4.3 Lateral Micro-Drift (Y-axis)

The Y-coordinate during contact exhibits slow drift from arm kinematics:

$$y(t) = y_{pass} + A_y \sin(2\pi f_y t + \phi_y)$$

where $A_y = 0.2$ mm, $f_y \sim \mathcal{U}(0.1, 0.3)$ Hz. Critically, $f_y$ is fixed per-trajectory (not per-timestep), producing coherent drift rather than white noise.

### 4.4 Pass Transition Modeling

At the end of each wiping pass, the end-effector transitions to the next pass start via a Bézier curve with:
- **Z-lift**: $\Delta z_{lift} \sim \mathcal{U}(1, 2)$ mm — modeling natural pressure release during direction change
- **X-overshoot**: $\Delta x_{over} \sim \mathcal{U}(1, 3)$ mm — modeling inertial lag at velocity reversal

The transition uses asymmetric control points:

$$P_1^{(z)} = z_c + \Delta z_{lift}, \quad P_2^{(z)} = z_c + \Delta z_{lift} \cdot \mathcal{U}(0.5, 1.0)$$

This produces a non-symmetric arch (faster rise than descent), consistent with biomechanical observations of direction reversal in wiping tasks.

---

## 5. Negative Sample Generation

Negative trajectories model specific contact failure modes while preserving kinematic naturalness in non-contact segments.

### 5.1 Excessive Contact Force ($z_{too\_low}$)

The contact height is reduced: $z_c' = z_c - \delta_z$, where $\delta_z \sim \mathcal{U}(3, 6)$ mm.

The entire trajectory is regenerated with the modified parameter, preserving smooth Bézier transitions. This ensures physical consistency—the approach naturally targets the lower Z, rather than exhibiting a discontinuous jump.

### 5.2 Insufficient Contact Force ($z_{too\_high}$)

Symmetrically: $z_c' = z_c + \delta_z$, where $\delta_z \sim \mathcal{U}(5, 10)$ mm.

The larger range reflects the nonlinear force-displacement relationship: insufficient contact requires larger positional deviation to produce distinguishable tactile signatures.

### 5.3 Unstable Contact ($z_{oscillate}$)

This mode models intermittent contact quality degradation. The perturbation is applied only to the contact segment of an otherwise valid positive trajectory:

$$z'(t) = z(t) + \eta(t) \cdot \psi(t)$$

where $\eta(t)$ is the perturbation signal and $\psi(t)$ is a smoothstep envelope.

**Perturbation signal** — multi-frequency superposition plus filtered random walk:

$$\eta(t) = \underbrace{\sum_{i=1}^{K} a_i \sin(2\pi f_i t + \phi_i)}_{\text{multi-modal oscillation}} + \underbrace{\text{LPF}\left[\sum_{s=0}^{t} \xi_s\right]}_{\text{stochastic drift}}$$

where:
- $K \sim \mathcal{U}\{3, 4, 5\}$ — number of frequency components
- $f_i \sim \mathcal{U}(0.3, 2.5)$ Hz — per-component frequency
- $a_i \sim \mathcal{U}(2, 6)$ mm — per-component amplitude
- $\xi_s \sim \mathcal{N}(0, 0.8 \text{ mm})$ — random walk increments
- LPF: moving average with kernel size $\lfloor N_{wipe}/20 \rfloor$, clipped to $\pm 3$ mm

**Smoothstep envelope** — ensures $C^1$ continuity at contact boundaries:

$$\psi(t) = \begin{cases} S(t/N_b) & t < N_b \\ 1 & N_b \leq t \leq N_w - N_b \\ S((N_w - t)/N_b) & t > N_w - N_b \end{cases}$$

where $S(x) = 3x^2 - 2x^3$ is the Hermite smoothstep and $N_b = \min(30, N_w/4)$.

---

## 6. Trajectory–Observation Alignment

The synthesized trajectory serves as the commanded action sequence. During real-robot execution at $f_c = 20$ Hz:

| Signal | Role | Source |
|--------|------|--------|
| $a_t^{cmd}$ (EEF target) | Action command | Trajectory file |
| $a_t^{joint}$ (joint angles) | Action ground-truth for training | Real-time state callback |
| $s_t^{proprio}$ (proprioception) | State observation | Real-time state callback |
| $s_t^{visual}$ (RGB images) | State observation | RealSense cameras (×2) |
| $s_t^{tactile}$ (tactile) | State observation | GelSlim sensors (×2, bilateral) |

The temporal alignment follows: command $a_t$ → execute → observe $s_t$. For policy learning, the input is $(s_t^{proprio}, s_t^{visual}, s_t^{tactile})$ and the prediction target is the future action chunk $\{a_{t+1}^{joint}, \ldots, a_{t+H}^{joint}\}$.

---

## 7. Multi-Modal Observation Space

Each timestep records:

$$\mathcal{O}_t = \left( q_t \in \mathbb{R}^7, \; I_t^{global} \in \mathbb{R}^{200\times266\times3}, \; I_t^{wrist} \in \mathbb{R}^{200\times266\times3}, \; \tau_t^{L}, \; \tau_t^{R} \right)$$

where the tactile observation per side is:

$$\tau_t = \left( I_t^{tac} \in \mathbb{R}^{240\times240\times3}, \; M_t \in \mathbb{R}^{9\times9\times2}, \; F_t \in \mathbb{R}^6 \right)$$

- $I_t^{tac}$: GelSlim raw image (deformation visualization)
- $M_t$: marker displacement field (9×9 grid, 2D offset per marker)
- $F_t$: 6-axis contact force/torque estimate

The bilateral design (left + right sensors) captures asymmetric contact patterns critical for quality assessment.

---

## 8. Statistical Validation

The synthesis parameters are validated against the reference dataset ($N=80$ human episodes):

| Metric | Human Data | Synthesized | Match |
|--------|-----------|-------------|-------|
| Contact Z mean | 125.6 ± 0.8 mm | 125.0 ± 1.0 mm | ✓ |
| Contact Z fluctuation (std) | 0.27 mm | ~0.25 mm | ✓ |
| Wipe speed | 9.4 ± 1.2 mm/s | 10.0 ± 1.0 mm/s | ✓ |
| X range | [270, 420] mm | [265, 425] mm | ✓ (superset) |
| Trajectory duration | 35–55 s | 40–50 s | ✓ |
| Jerk bound | < 500 mm/s³ | < 300 mm/s³ | ✓ (smoother) |

---

## 9. Dataset Composition

| Category | Count | Contact Z | Failure Signature |
|----------|-------|-----------|-------------------|
| Positive | 150 | 124–126 mm | Stable contact, $\sigma_z < 0.5$ mm |
| Neg: z_oscillate | 60 | 125 ± 2–6 mm (varying) | Irregular Z perturbation, $\sigma_z > 3$ mm |
| Neg: z_too_high | 60 | 130–135 mm | Insufficient contact force |
| Neg: z_too_low | 60 | 119–122 mm | Excessive contact force |

**Total**: 330 trajectories → 330 episodes after real-robot execution (~5.5 hours at ~1 min/episode).

---

## 10. Implementation Notes

- **Reproducibility**: Each trajectory is seeded; given the same seed and parameters, identical output is guaranteed.
- **Safety**: Dry-run mode validates workspace bounds, velocity limits, and force thresholds before any physical execution.
- **Extensibility**: The `WipeTrajectoryGenerator` class accepts arbitrary parameter overrides via CLI, enabling rapid exploration of new parameter regimes without code modification.
