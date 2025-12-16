# Work In Progress: Moving Subdomain Boundaries

## Status: Not Working

The implementation compiles and runs without crashing, but the physics results are incorrect. The algorithm for estimating subdomain edge velocities at shocks needs improvement.

## What Was Implemented

Based on `spec.md`, the following changes were made to `src/blast.cpp`:

### 1. Element Structure in `patch_t`
- `Ne` - number of elements (subdomains)
- `Nz` - zones per element
- `edges` - element boundary positions (size Ne+1)
- `v_edge` - element boundary velocities (size Ne+1)
- `edge_type` - classification: 0=neither, 1=contact, 2=shock (size Ne+1)
- `fhat_edge` - flux density at element boundaries (size Ne+1)

### 2. Element-Aware Geometry Methods
Added to `patch_t`: `face_radius()`, `face_velocity()`, `face_area()`, `cell_volume()`, `cell_radius()` with guard cell handling via linear extrapolation.

### 3. Configuration Tolerances
Added to `config_t`:
- `tol_contact_up` - velocity/pressure matching tolerance for contacts
- `tol_contact_rho` - minimum density jump for contacts
- `tol_shock` - shock speed agreement tolerance

### 4. Discontinuity Detection Helpers
- `reldiff(a, b)` - relative difference
- `is_contact(pL, pR, tol_up, tol_rho)` - checks if boundary is a contact
- `is_shock(uL, uR, fL, fR, v_shock, tol)` - checks if boundary is a shock

### 5. New Pipeline Stage
`classify_element_boundaries_t` - classifies each element boundary and computes edge velocities and Rankine-Hugoniot fluxes.

### 6. Updated Pipeline Stages
- `compute_fluxes_t` - applies flux overrides at element boundaries
- `update_conserved_t` - moves element boundaries each timestep
- `local_dt_t` - uses minimum element width for CFL

### 7. Initial Conditions
- `wind` and `blast_wave` use 2 elements
- All others use 1 element

### 8. Serialization
`edges` vector is serialized for restarts.

## Known Issues

1. **Shock velocity estimation is incorrect** - The current algorithm uses Rankine-Hugoniot jump conditions to infer shock speed from `dF/dU`, but this doesn't work well in practice. The spec will be updated with a better algorithm.

2. **Contact detection may be too sensitive** - The tolerance-based detection may not reliably identify contacts.

## Next Steps

- Update `spec.md` with improved algorithm for subdomain edge velocity estimation at shocks
- Revisit discontinuity detection logic
- Add diagnostic output to debug edge classification
