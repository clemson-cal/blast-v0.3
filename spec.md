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

**Step 1: Check density continuity**
- If `reldiff(ρL, ρR) < tol_rho`: density is continuous → **neither shock nor contact** (edge_type = 0)

**Step 2: Check four-velocity continuity** (given density is discontinuous)
- If `reldiff(uL, uR) > tol_u`: four-velocity is discontinuous → **shock** (edge_type = 2)
- If `reldiff(uL, uR) < tol_u`: four-velocity is continuous → **contact** (edge_type = 1)

### Edge Velocity Computation

**Shock (edge_type = 2):**
```cpp
gamma_rel = sqrt(1.0 + u_u * u_u) * sqrt(1.0 + u_d * u_d) - u_u * u_d;
gamma_index = 4.0 / 3.0;
gamma_shock = sqrt((gamma_rel + 1.0) * pow(gamma_index * (gamma_rel - 1.0) + 1.0, 2)
                   / (gamma_index * (2.0 - gamma_index) * (gamma_rel - 1.0) + 2.0));
beta_shock = -sqrt(1.0 - 1.0 / (gamma_shock * gamma_shock));
beta_d = u_d / sqrt(1.0 + u_d * u_d);
v_edge[k] = (beta_d + beta_shock) / (1.0 + beta_d * beta_shock);
```
where `u_u` is upstream four-velocity and `u_d` is downstream four-velocity.

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
