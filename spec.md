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

---

## Alternative Prescription: Shocks as Domain Boundaries (Four-State)

For the `four_state` initial condition, an alternative approach is to evolve only the shocked regions with the shocks serving as the physical domain boundaries.

### Setup

- **Patches**: 2 (not 4)
  - Patch 0: Shocked left material (region 3), domain [r_rs, r_cd]
  - Patch 1: Shocked right material (region 2), domain [r_cd, r_fs]
- **Unshocked material**: Not on the grid (regions 1 and 4 are external)
- **Domain boundaries**: ARE the shocks (r_rs and r_fs)

### Initialization

At `tstart`, compute analytical shock positions and velocities:

```cpp
// Solve two-shock Riemann problem
auto sol = riemann::solve_two_shock(d_L, u_L, d_R, u_R);
auto [v_rs, v_cd, v_fs] = riemann::compute_discontinuity_velocities({d_L, u_L, d_R, u_R}, sol);

// Discontinuity positions at tstart (launched from r=1.0 at t=0)
double r_rs = 1.0 + v_rs * tstart;  // reverse shock
double r_cd = 1.0 + v_cd * tstart;  // contact discontinuity
double r_fs = 1.0 + v_fs * tstart;  // forward shock

// Patch 0: [r_rs, r_cd]
patch[0].r0 = r_rs;
patch[0].r1 = r_cd;
patch[0].v0 = v_rs;  // reverse shock velocity
patch[0].v1 = v_cd;  // contact velocity
patch[0].e0 = edge_type::shock;
patch[0].e1 = edge_type::contact;

// Patch 1: [r_cd, r_fs]
patch[1].r0 = r_cd;
patch[1].r1 = r_fs;
patch[1].v0 = v_cd;  // contact velocity
patch[1].v1 = v_fs;  // forward shock velocity
patch[1].e0 = edge_type::contact;
patch[1].e1 = edge_type::shock;

// Initialize with shocked states from two-shock solution
// Patch 0: d3, u, p (shocked left)
// Patch 1: d2, u, p (shocked right)
```

### Evolution

**Edge velocities**: Preserve analytical values throughout evolution. Do NOT override with dynamically computed values from `classify_patch_edges_t`.

**Discontinuity fluxes**: Compute using interior states and analytical velocities:
```cpp
if (ic == initial_condition::four_state) {
    // Skip dynamic classification, use analytical velocities
    if (p.e0 != edge_type::generic) {
        auto pR = p.prim[i0];
        auto uR = prim_to_cons(pR);
        auto fR = prim_and_cons_to_flux(pR, uR);
        p.discontinuity_flux_l = fR - uR * p.v0;  // p.v0 is analytical
    }

    if (p.e1 != edge_type::generic) {
        auto pL = p.prim[i1 - 1];
        auto uL = prim_to_cons(pL);
        auto fL = prim_and_cons_to_flux(pL, uL);
        p.discontinuity_flux_r = fL - uL * p.v1;  // p.v1 is analytical
    }
}
```

### Rationale

This approach treats the shocks as true boundaries of the computational domain rather than as internal features. The unshocked material is handled analytically (positions and velocities from the two-shock Riemann solution) rather than being explicitly represented on the grid.

**Advantages**:
- Avoids very small patches at early times
- Cleaner separation between shocked and unshocked regions
- Domain boundaries have clear physical meaning

**Challenges**:
- Domain boundaries are moving (shocks)
- Need to ensure proper flux treatment at shock boundaries
- CFL constraint may be severe if patches remain narrow
