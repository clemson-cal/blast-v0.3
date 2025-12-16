# Blast v0.3 Specification Summary

A modification to the existing 1D SRHD code that decomposes the domain into moving subdomains (elements) whose boundaries track shocks and contacts.

> **TODO**: The shock velocity estimation algorithm (using `dF/dU` from Rankine-Hugoniot) does not work reliably in practice. This spec will be updated with a better algorithm for estimating subdomain edge velocities at shocks.

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

### Contact Condition
A contact exists if **all** of:
- `reldiff(uL, uR) <= tol_up`
- `reldiff(pL, pR) <= tol_up`
- `reldiff(ρL, ρR) >= tol_rho_jump`

```cpp
v = beta(PL);
// the flux across the discontinuity is:
fhat = FL - UL * v = FR - UR * v // these are close within tol (can use average)
```

### Shock Condition
```cpp
dU = UR - UL;
dF = FR - FL;

sD = dF[D] / dU[D];  // inferred speeds
sS = dF[S] / dU[S];
sE = dF[E] / dU[E];

// Require all speeds agree within tolerance
v = (sD + sS + sE) / 3;
// the flux across the discontinuity is:
fhat = FL - UL * v = FR - UR * v // these are close within tol (can use average)
```

### Classification
```cpp
if (contact_passes)      { edge_type[k] = 1; v_edge[k] = v; }
else if (shock_passes)   { edge_type[k] = 2; v_edge[k] = v; }
else                     { edge_type[k] = 0; v_edge[k] = 1/2 * (vl + vr); } // <- FIXED; vl and vr are the fluid speeds in zones adjacent to the subdomain edge
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
