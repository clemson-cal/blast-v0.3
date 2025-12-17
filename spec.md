# Blast v0.3 Specification Summary

A modification to the existing 1D SRHD code that decomposes the domain into moving subdomains (elements) whose boundaries track shocks and contacts.

---

## Subdomain Structure

| Setup | Ne (elements) |
|-------|---------------|
| wind, blast_wave | 2 |
| uniform, sod | 1 |

- Total zones: `num_zones = Ne * Nz`
- Constraint: `num_zones % Ne == 0`

---

## New Fields in `patch_t`

```cpp
unsigned Ne, Nz;
std::vector<double> edges;     // size Ne+1
std::vector<double> v_edge;    // size Ne+1
std::vector<int> edge_type;    // 0 = neither, 1 = contact, 2 = shock
```

---

## Geometry

Face radius for global face index `f`:

```cpp
// Determine element index
if (f == num_zones) {
    e = Ne - 1;
    j = Nz;
} else {
    e = f / Nz;
    j = f - e * Nz;
}

// Compute face radius
r_face(f) = edges[e] + (j / Nz) * (edges[e+1] - edges[e])
```

Derived quantities:
```cpp
face_area(f)   = four_pi * r_face(f)^2
cell_volume(i) = four_pi/3 * (r_face(i+1)^3 - r_face(i)^3)
cell_radius(i) = 0.5 * (r_face(i) + r_face(i+1))
```

---

## Discontinuity Detection

All checks use tolerance parameters for numerical robustness.

### Classification Algorithm

**Step 1: Check shock jump conditions (Rankine-Hugoniot)**
- Compute shock velocity `v_s` from the states using relativistic jump relations
- Compute flux in shock frame on both sides: `flux_L = F_L - v_s * U_L`, `flux_R = F_R - v_s * U_R`
- If `reldiff(flux_L, flux_R) < shock_tol`: states satisfy shock conditions → **shock** (edge_type = 2)

**Step 2: Check contact jump conditions**
- If `reldiff(v_L, v_R) < contact_tol` AND `reldiff(p_L, p_R) < contact_tol`: velocity and pressure are continuous → **contact** (edge_type = 1)

**Step 3: Otherwise**
- Neither shock nor contact conditions satisfied → **generic** (edge_type = 0)

### Edge Velocity Computation

**Shock (edge_type = 2):**
```cpp
// Determine upstream (unshocked) vs downstream (shocked) using pressure
double u_u, u_d, p_u, p_d;
double sign_shock;
if (PL[2] > PR[2]) {
    // Left is downstream (shocked), right is upstream (unshocked)
    u_d = PL[1];  // four-velocity from primitive (gamma*beta)
    u_u = PR[1];
    p_d = PL[2];
    p_u = PR[2];
    sign_shock = -1.0;  // shock propagates from left to right (negative direction)
} else {
    // Right is downstream (shocked), left is upstream (unshocked)
    u_d = PR[1];
    u_u = PL[1];
    p_d = PR[2];
    p_u = PL[2];
    sign_shock = +1.0;  // shock propagates from right to left (positive direction)
}

gamma_rel = sqrt(1.0 + u_u * u_u) * sqrt(1.0 + u_d * u_d) - u_u * u_d;
gamma_index = 4.0 / 3.0;
gamma_shock = sqrt((gamma_rel + 1.0) * pow(gamma_index * (gamma_rel - 1.0) + 1.0, 2)
                   / (gamma_index * (2.0 - gamma_index) * (gamma_rel - 1.0) + 2.0));
beta_shock = sign_shock * sqrt(1.0 - 1.0 / (gamma_shock * gamma_shock));
beta_u = u_u / sqrt(1.0 + u_u * u_u);
v_edge[k] = (beta_u + beta_shock) / (1.0 + beta_u * beta_shock);
```
The shock velocity `beta_shock` is in the upstream rest frame and is boosted to the lab frame using `beta_u`.
Upstream (unshocked) vs downstream (shocked) regions are determined by comparing pressures.

**Contact (edge_type = 1):**
```cpp
v_edge[k] = 0.5 * (beta(PL) + beta(PR));  // average of adjacent fluid velocities
```

**Neither (edge_type = 0):**
```cpp
v_edge[k] = 0.5 * (beta(PL) + beta(PR));  // same as contact case
```

### Flux at Discontinuities

For edges with `edge_type > 0`:
```cpp
// Use interior-biased gradients relative to the boundary
fhat_density[k] = FL - v_edge[k] * UL;  // or FR - v_edge[k] * UR (should be close)
```

---

## Face Velocities

Linear interpolation within element `e`:

```cpp
vL = v_edge[e];
vR = v_edge[e+1];
alpha = (r_face(f) - edges[e]) / (edges[e+1] - edges[e]);
v_face(f) = (1 - alpha) * vL + alpha * vR;
```

---

## Flux Computation

For each face:
```cpp
// Compute HLLE with moving face
flux_density = riemann_hlle(PL, PR, UL, UR, v_face(f));
F_area = flux_density * face_area(f);
```

At element boundaries `f = k * Nz`:
```cpp
if (edge_type[k] == 1 || edge_type[k] == 2) {
    fhat[f] = fhat_density[k] * face_area(f);  // override
}
```

---

## Conserved Update & Mesh Motion

```cpp
// Update conserved variables
cons[i] -= (fhat[i+1] - fhat[i]) * dt;
cons[i] += spherical_source * dt;

// Move element boundaries
for (k = 0; k <= Ne; k++) {
    edges[k] += dt * v_edge[k];
}

// Enforce monotonicity
// edges[k+1] > edges[k] + min_width  (min_width ~ 1e-12)
```

---

## CFL Handling

```cpp
dr_eff = min_k(edges[k+1] - edges[k]) / Nz;
grid.dr = dr_eff;
dt = cfl * grid.dr / max_wavespeed;
```

---

## Pipeline Stages

1. Gradient computation
2. **NEW: Classify element boundaries** → compute `v_edge`, `edge_type`, `fhat_density`
3. Flux computation (with `v_face` and boundary overrides)
4. Conserved update + mesh motion + update `grid.dr`

---

## Serialization

Only `edges` must be serialized for restarts. Derived quantities (`v_edge`, `edge_type`, `fhat`) are recomputed.
