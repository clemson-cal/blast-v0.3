/**
================================================================================
Copyright 2023 - 2025, Jonathan Zrake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
================================================================================
*/

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include "mist/core.hpp"
#include "mist/driver/physics_impl.hpp"
#include "mist/driver/repl_session.hpp"
#include "mist/driver/socket_session.hpp"
#include "mist/ndarray.hpp"
#include "mist/pipeline.hpp"
#include "mist/serialize.hpp"
#include "riemann_two_shock.hpp"

using namespace mist;

// =============================================================================
// Type aliases
// =============================================================================

using cons_t = dvec_t<3>;  // conserved variables: (D, S, tau)
using prim_t = dvec_t<3>;  // primitive variables: (rho, u, p)

template<std::ranges::range R>
auto to_vector(R&& r) {
    auto v = std::vector<std::ranges::range_value_t<R>>{};
    for (auto&& e : r) {
        v.push_back(std::forward<decltype(e)>(e));
    }
    return v;
}

// =============================================================================
// Math utility functions
// =============================================================================

static constexpr double gamma_law = 4.0 / 3.0;

static inline auto min2(double a, double b) -> double {
    return a < b ? a : b;
}

static inline auto max2(double a, double b) -> double {
    return a > b ? a : b;
}

static inline auto min3(double a, double b, double c) -> double {
    return min2(a, min2(b, c));
}

static inline auto max3(double a, double b, double c) -> double {
    return max2(a, max2(b, c));
}

static inline auto sign(double x) -> double {
    return std::copysign(1.0, x);
}

static inline auto minabs(double a, double b, double c) -> double {
    return min3(std::fabs(a), std::fabs(b), std::fabs(c));
}

static inline auto plm_minmod(double yl, double yc, double yr, double plm_theta) -> double {
    auto a = (yc - yl) * plm_theta;
    auto b = (yr - yl) * 0.5;
    auto c = (yr - yc) * plm_theta;
    return 0.25 * std::fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

template<typename T, std::size_t N>
static inline auto plm_gradient(vec_t<T, N> yl, vec_t<T, N> yc, vec_t<T, N> yr, double plm_theta) -> vec_t<T, N> {
    auto result = vec_t<T, N>{};
    for (std::size_t q = 0; q < N; ++q) {
        result[q] = plm_minmod(yl[q], yc[q], yr[q], plm_theta);
    }
    return result;
}

// =============================================================================
// SR Hydrodynamics functions
// =============================================================================

static constexpr double four_pi = 1.0;

static auto gamma_beta_squared(prim_t p) -> double {
    return p[1] * p[1];
}

static auto momentum_squared(cons_t u) -> double {
    return u[1] * u[1];
}

static auto lorentz_factor(prim_t p) -> double {
    return std::sqrt(1.0 + gamma_beta_squared(p));
}

static auto beta(prim_t p) -> double {
    return p[1] / lorentz_factor(p);
}

static auto enthalpy_density(prim_t p) -> double {
    auto rho = p[0];
    auto pre = p[2];
    return rho + pre * (1.0 + 1.0 / (gamma_law - 1.0));
}

static auto prim_to_cons(prim_t p) -> cons_t {
    auto rho = p[0];
    auto pre = p[2];
    auto w = lorentz_factor(p);
    auto h = enthalpy_density(p) / rho;
    auto m = rho * w;
    auto u = cons_t{};
    u[0] = m;
    u[1] = m * (h * p[1]);
    u[2] = m * (h * w - 1.0) - pre;
    return u;
}

static auto cons_to_prim(cons_t cons, double p = 0.0) -> prim_t {
    auto newton_iter_max = 50;
    auto error_tolerance = 1e-12 * (cons[0] + cons[2]);
    auto gm = gamma_law;
    auto m = cons[0];
    auto tau = cons[2];
    auto ss = momentum_squared(cons);
    auto w0 = 0.0;

    for (int n = 0; n < newton_iter_max; ++n) {
        auto et = tau + p + m;
        auto b2 = min2(ss / et / et, 1.0 - 1e-10);
        auto w2 = 1.0 / (1.0 - b2);
        auto w = std::sqrt(w2);
        auto d = m / w;
        auto de = (tau + m * (1.0 - w) + p * (1.0 - w2)) / w2;
        auto dh = d + de + p;
        auto a2 = gm * p / dh;
        auto g = b2 * a2 - 1.0;
        auto f = de * (gm - 1.0) - p;

        if (std::fabs(f) < error_tolerance) {
            w0 = w;
            break;
        }
        p -= f / g;
    }
    return prim_t{m / w0, w0 * cons[1] / (tau + m + p), p};
}

static auto prim_and_cons_to_flux(prim_t p, cons_t u) -> cons_t {
    auto pre = p[2];
    auto vn = beta(p);
    auto f = cons_t{};
    f[0] = vn * u[0];
    f[1] = vn * u[1] + pre;
    f[2] = vn * u[2] + pre * vn;
    return f;
}

static auto sound_speed_squared(prim_t p) -> double {
    auto pre = p[2];
    auto rho_h = enthalpy_density(p);
    return gamma_law * pre / rho_h;
}

static auto outer_wavespeeds(prim_t p) -> dvec_t<2> {
    auto a2 = sound_speed_squared(p);
    auto uu = gamma_beta_squared(p);
    auto vn = beta(p);
    auto g2 = 1.0 + uu;
    auto s2 = a2 / g2 / (1.0 - a2);
    auto v2 = vn * vn;
    auto k0 = std::sqrt(s2 * (1.0 - v2 + s2));
    return dvec(vn - k0, vn + k0) / (1.0 + s2);
}

static auto riemann_hlle(prim_t pl, prim_t pr, cons_t ul, cons_t ur, double v_face = 0.0) -> cons_t {
    auto fl = prim_and_cons_to_flux(pl, ul);
    auto fr = prim_and_cons_to_flux(pr, ur);

    auto cl = std::sqrt(sound_speed_squared(pl));
    auto cr = std::sqrt(sound_speed_squared(pr));
    auto vl = beta(pl);
    auto vr = beta(pr);
    auto alm = (vl - cl) / (1.0 - vl * cl);
    auto alp = (vl + cl) / (1.0 + vl * cl);
    auto arm = (vr - cr) / (1.0 - vr * cr);
    auto arp = (vr + cr) / (1.0 + vr * cr);
    auto am = min2(alm, arm);
    auto ap = max2(alp, arp);

    if (v_face < am) {
        return fl - ul * v_face;
    }
    if (v_face > ap) {
        return fr - ur * v_face;
    }
    auto u_hll = (ur * ap - ul * am + (fl - fr)) / (ap - am);
    auto f_hll = (fl * ap - fr * am - (ul - ur) * ap * am) / (ap - am);
    return f_hll - u_hll * v_face;
}

static auto riemann_hllc(prim_t pl, prim_t pr, cons_t ul, cons_t ur, double v_face = 0.0) -> cons_t {
    auto fl = prim_and_cons_to_flux(pl, ul);
    auto fr = prim_and_cons_to_flux(pr, ur);

    auto ws_l = outer_wavespeeds(pl);
    auto ws_r = outer_wavespeeds(pr);
    auto alm = ws_l[0];
    auto alp = ws_l[1];
    auto arm = ws_r[0];
    auto arp = ws_r[1];
    auto ar = max2(alp, arp);
    auto al = min2(alm, arm);

    // Equations (9) and (11) from Mignone & Bodo 2005
    auto u_hll = (ur * ar - ul * al + (fl - fr))           / (ar - al);
    auto f_hll = (fl * ar - fr * al - (ul - ur) * ar * al) / (ar - al);

    auto discriminant = [](double a, double b, double c) -> double {
        return b * b - 4.0 * a * c;
    };

    auto quadratic_root = [&discriminant](double a, double b, double c) -> double {
        auto d = discriminant(a, b, c);
        if (d < 0.0) {
            return 0.0;
        } else if (std::fabs(a) < 1e-8) {
            return -c / b;
        } else {
            return (-b - std::sqrt(d)) / 2.0 / a;
        }
    };

    // Equation (18) for a-star and p-star
    auto a_star_and_p_star = [&]() -> std::pair<double, double> {
        // Mignone defines total energy to include rest mass
        auto ue_hll = u_hll[2] + u_hll[0];
        auto fe_hll = f_hll[2] + f_hll[0];
        auto um_hll = u_hll[1];
        auto fm_hll = f_hll[1];
        auto a_star = quadratic_root(fe_hll, -fm_hll - ue_hll, um_hll);
        auto p_star = -fe_hll * a_star + fm_hll;
        if (std::isnan(a_star)) {
            std::cerr << "a* is NaN, pl = [" << pl[0] << ", " << pl[1] << ", " << pl[2]
                      << "], pr = [" << pr[0] << ", " << pr[1] << ", " << pr[2] << "]\n";
        }
        return {a_star, p_star};
    };

    // Equations (16)
    auto star_state_flux = [](cons_t u, cons_t f, prim_t p, double a, double vface, double a_star, double p_star) -> cons_t {
        auto e = u[2] + u[0];
        auto m = u[1];
        auto v = beta(p);
        auto es = (e * (a - v) + p_star * a_star - p[2] * v) / (a - a_star);
        auto ms = (m * (a - v) + p_star          - p[2])     / (a - a_star);
        auto ds = u[0] * (a - v)                             / (a - a_star);
        auto us = cons_t{ds, ms, es - ds};
        auto fs = f + (us - u) * a;
        return fs - us * vface;
    };

    if (v_face <= al) {
        return fl - ul * v_face;
    } else if (v_face >= ar) {
        return fr - ur * v_face;
    } else {
        auto [a_star, p_star] = a_star_and_p_star();
        if (v_face <= a_star) {
            return star_state_flux(ul, fl, pl, al, v_face, a_star, p_star);
        } else {
            return star_state_flux(ur, fr, pr, ar, v_face, a_star, p_star);
        }
    }
}

static auto max_wavespeed(prim_t p) -> double {
    auto ws = outer_wavespeeds(p);
    return max2(std::fabs(ws[0]), std::fabs(ws[1]));
}

static auto spherical_geometry_source_terms(prim_t p, double r0, double r1) -> cons_t {
    // Eqn A8 in Zhang & MacFadyen (2006), integrated over the spherical shell
    // between r0 and r1, and specializing to radial velocity only.
    // Source = 4π p (r1² - r0²) to match the area-integrated fluxes
    auto pg = p[2];
    auto dr2 = std::pow(r1, 2) - std::pow(r0, 2);
    auto srdot = four_pi * pg * dr2;
    return cons_t{0.0, srdot, 0.0};
}

// =============================================================================
// Discontinuity detection helpers
// =============================================================================

static auto reldiff(double a, double b) -> double {
    return std::fabs(a - b) / (0.5 * (std::fabs(a) + std::fabs(b)) + 1e-14);
}

// New spec: Step 1 - check density continuity
static auto has_density_jump(prim_t pL, prim_t pR, double tol_rho) -> bool {
    return reldiff(pL[0], pR[0]) >= tol_rho;
}

// New spec: Step 2 - check four-velocity continuity (given density is discontinuous)
static auto has_velocity_jump(prim_t pL, prim_t pR, double tol_u) -> bool {
    return reldiff(pL[1], pR[1]) > tol_u;
}

// Compute shock velocity using relativistic jump relations
static auto compute_shock_velocity(prim_t pL, prim_t pR) -> double {
    constexpr double gamma_index = gamma_law;  // 4/3

    // Determine upstream (unshocked) vs downstream (shocked) using pressure
    double u_u, u_d, p_u, p_d;
    double sign_shock;

    if (pL[2] > pR[2]) {
        // Left is downstream (shocked), right is upstream (unshocked)
        u_d = pL[1];
        u_u = pR[1];
        p_d = pL[2];
        p_u = pR[2];
        sign_shock = -1.0;  // shock propagates left to right
    } else {
        // Right is downstream (shocked), left is upstream (unshocked)
        u_d = pR[1];
        u_u = pL[1];
        p_d = pR[2];
        p_u = pL[2];
        sign_shock = +1.0;  // shock propagates right to left
    }

    // Relative Lorentz factor
    double gamma_rel = std::sqrt(1.0 + u_u * u_u) * std::sqrt(1.0 + u_d * u_d) - u_u * u_d;

    // Shock Lorentz factor (in upstream rest frame)
    double gamma_shock = std::sqrt((gamma_rel + 1.0) * std::pow(gamma_index * (gamma_rel - 1.0) + 1.0, 2.0)
                                   / (gamma_index * (2.0 - gamma_index) * (gamma_rel - 1.0) + 2.0));

    // Shock velocity in upstream rest frame
    double beta_shock = sign_shock * std::sqrt(1.0 - 1.0 / (gamma_shock * gamma_shock));

    // Upstream velocity in lab frame
    double beta_u = u_u / std::sqrt(1.0 + u_u * u_u);

    // Boost shock velocity to lab frame using relativistic velocity addition
    double v_edge = (beta_u + beta_shock) / (1.0 + beta_u * beta_shock);

    return v_edge;
}

// =============================================================================
// Riemann solver types
// =============================================================================

enum class riemann_solver {
    hlle,
    hllc,
    two_shock,  // not implemented yet
    exact       // not implemented yet
};

auto to_string(riemann_solver rs) -> const char* {
    switch (rs) {
        case riemann_solver::hlle: return "hlle";
        case riemann_solver::hllc: return "hllc";
        case riemann_solver::two_shock: return "two_shock";
        case riemann_solver::exact: return "exact";
    }
    return "unknown";
}

auto from_string(std::type_identity<riemann_solver>, const std::string& s) -> riemann_solver {
    if (s == "hlle") return riemann_solver::hlle;
    if (s == "hllc") return riemann_solver::hllc;
    if (s == "two_shock") return riemann_solver::two_shock;
    if (s == "exact") return riemann_solver::exact;
    throw std::runtime_error("unknown riemann_solver: " + s);
}

// =============================================================================
// Geometry
// =============================================================================

enum class geometry {
    planar,
    spherical
};

auto to_string(geometry g) -> const char* {
    switch (g) {
        case geometry::planar: return "planar";
        case geometry::spherical: return "spherical";
    }
    return "unknown";
}

auto from_string(std::type_identity<geometry>, const std::string& s) -> geometry {
    if (s == "planar") return geometry::planar;
    if (s == "spherical") return geometry::spherical;
    throw std::runtime_error("unknown geometry: " + s);
}

struct grid_t {
    double r0_initial = 0.0;  // inner radius at t=0
    double r1_initial = 0.0;  // outer radius at t=0
    double v_inner = 0.0;     // inner boundary velocity
    double v_outer = 0.0;     // outer boundary velocity
    unsigned num_zones = 0;
    geometry geom = geometry::spherical;

    auto inner_radius(double t) const -> double {
        return r0_initial + v_inner * t;
    }

    auto outer_radius(double t) const -> double {
        return r1_initial + v_outer * t;
    }

    auto dr(double t) const -> double {
        return (outer_radius(t) - inner_radius(t)) / num_zones;
    }

    auto face_radius(int i, double t) const -> double {
        return inner_radius(t) + i * dr(t);
    }

    auto cell_radius(int i, double t) const -> double {
        return inner_radius(t) + (i + 0.5) * dr(t);
    }

    auto face_velocity(int i) const -> double {
        return v_inner + (v_outer - v_inner) * i / num_zones;
    }

    auto face_area(int i, double t) const -> double {
        if (geom == geometry::planar) {
            return 1.0;
        } else {
            auto r = face_radius(i, t);
            return four_pi * r * r;
        }
    }

    auto cell_volume(int i, double t) const -> double {
        if (geom == geometry::planar) {
            return dr(t);  // Volume is just dx for planar
        } else {
            auto rl = face_radius(i, t);
            auto rr = face_radius(i + 1, t);
            return four_pi / 3.0 * (rr * rr * rr - rl * rl * rl);
        }
    }
};

// =============================================================================
// Initial conditions
// =============================================================================

enum class initial_condition {
    sod,
    blast_wave,
    wind,
    uniform,
    four_state,
    mignone_bodo,
};

auto to_string(initial_condition ic) -> const char* {
    switch (ic) {
        case initial_condition::sod: return "sod";
        case initial_condition::blast_wave: return "blast_wave";
        case initial_condition::wind: return "wind";
        case initial_condition::uniform: return "uniform";
        case initial_condition::four_state: return "four_state";
        case initial_condition::mignone_bodo: return "mignone_bodo";
    }
    return "unknown";
}

auto from_string(std::type_identity<initial_condition>, const std::string& s) -> initial_condition {
    if (s == "sod") return initial_condition::sod;
    if (s == "blast_wave") return initial_condition::blast_wave;
    if (s == "wind") return initial_condition::wind;
    if (s == "uniform") return initial_condition::uniform;
    if (s == "four_state") return initial_condition::four_state;
    if (s == "mignone_bodo") return initial_condition::mignone_bodo;
    throw std::runtime_error("unknown initial condition: " + s);
}

static auto num_elements_for_ic(initial_condition ic) -> unsigned {
    switch (ic) {
        case initial_condition::wind:
        case initial_condition::blast_wave:
            return 2;
        default:
            return 1;
    }
}

static auto initial_primitive(initial_condition ic, double r, double tstart = 0.0) -> prim_t {
    switch (ic) {
        case initial_condition::sod:
            if (r < 1.0) {
                return prim_t{1.0, 0.0, 1.0};
            } else {
                return prim_t{0.125, 0.0, 0.1};
            }
        case initial_condition::blast_wave:
            if (r < 1.0) {
                return prim_t{1.0, 0.0, 1000.0};
            } else {
                return prim_t{1.0, 0.0, 0.01};
            }
        case initial_condition::wind: {
            auto rho = 1.0 / (r * r);
            auto u = 1.0;  // gamma-beta = 1.0
            auto pre = 1e-6 * rho;
            return prim_t{rho, u, pre};
        }
        case initial_condition::uniform:
            return prim_t{1.0, 0.0, 1.0};
        case initial_condition::four_state: {
            // Converging cold flow: left moves right, right moves left
            constexpr double dl = 10.0;
            constexpr double ul = 10.0;   // four-velocity, moving right
            constexpr double dr = 1.0;
            constexpr double ur = 0.0;  // four-velocity, moving left
            constexpr double p_cold = 1e-6;

            // Solve two-shock Riemann problem
            auto sol = riemann::solve_two_shock(dl, ul, dr, ur);

            // Compute shock and contact velocities
            auto [v_rs, v_fs] = riemann::compute_shock_velocities({dl, ul, dr, ur}, sol);
            double g_contact = std::sqrt(1.0 + sol.u * sol.u);
            double v_cd = sol.u / g_contact;

            // Positions at tstart (shocks launched from r=1.0)
            double r_rs = 1.0 + v_rs * tstart;  // reverse shock position
            double r_cd = 1.0 + v_cd * tstart;  // contact position
            double r_fs = 1.0 + v_fs * tstart;  // forward shock position

            // Determine which region r falls into
            if (r < r_rs) {
                // Region 4: original left state
                return prim_t{dl, ul, p_cold * dl};
            } else if (r < r_cd) {
                // Region 3: shocked left material
                return prim_t{sol.d3, sol.u, sol.p};
            } else if (r < r_fs) {
                // Region 2: shocked right material
                return prim_t{sol.d2, sol.u, sol.p};
            } else {
                // Region 1: original right state
                return prim_t{dr, ur, p_cold * dr};
            }
        }
        case initial_condition::mignone_bodo:
            if (r < 5.0) {
                return prim_t{1.0, 2.065, 1.0};
            } else {
                return prim_t{1.0, 0.0, 10.0};
            }
    }
    assert(false);
}

// =============================================================================
// Boundary conditions
// =============================================================================

enum class boundary_condition {
    outflow,
    inflow,
    reflecting
};

auto to_string(boundary_condition bc) -> const char* {
    switch (bc) {
        case boundary_condition::outflow: return "outflow";
        case boundary_condition::inflow: return "inflow";
        case boundary_condition::reflecting: return "reflecting";
    }
    return "unknown";
}

auto from_string(std::type_identity<boundary_condition>, const std::string& s) -> boundary_condition {
    if (s == "outflow") return boundary_condition::outflow;
    if (s == "inflow") return boundary_condition::inflow;
    if (s == "reflecting") return boundary_condition::reflecting;
    throw std::runtime_error("unknown boundary condition: " + s);
}

// =============================================================================
// Patch - unified context that flows through pipeline
// =============================================================================

struct patch_t {
    index_space_t<1> interior;

    grid_t grid;
    double dt = 0.0;
    double time = 0.0;
    double time_rk = 0.0;
    double plm_theta = 1.5;

    // Element structure for moving subdomain boundaries
    unsigned Ne = 1;                      // number of elements
    unsigned Nz = 0;                      // zones per element
    std::vector<double> edges;            // size Ne+1, element boundary positions
    std::vector<double> v_edge;           // size Ne+1, element boundary velocities
    std::vector<int> edge_type;           // size Ne+1, 0=neither, 1=contact, 2=shock
    std::vector<cons_t> fhat_edge;        // size Ne+1, flux density at element boundaries

    cached_t<cons_t, 1> cons;
    cached_t<cons_t, 1> cons_rk;  // RK cached state
    mutable cached_t<prim_t, 1> prim;  // primitive variables at cell centers
    cached_t<prim_t, 1> grad;     // PLM gradients at cell centers
    cached_t<cons_t, 1> fhat;     // Godunov fluxes at faces

    patch_t() = default;

    patch_t(index_space_t<1> s)
        : interior(s)
        , cons(cache(fill(expand(s, 2), cons_t{0.0, 0.0, 0.0}), memory::host, exec::cpu))
        , cons_rk(cache(fill(expand(s, 2), cons_t{0.0, 0.0, 0.0}), memory::host, exec::cpu))
        , prim(cache(fill(expand(s, 2), prim_t{0.0, 0.0, 0.0}), memory::host, exec::cpu))
        , grad(cache(fill(expand(s, 1), prim_t{0.0, 0.0, 0.0}), memory::host, exec::cpu))
        , fhat(cache(fill(index_space(start(s), shape(s) + uvec(1)), cons_t{0.0, 0.0, 0.0}), memory::host, exec::cpu))
    {
    }

    // Element-aware geometry methods (handles guard cells via extrapolation)
    auto face_radius(int f) const -> double {
        // f is global face index; interior range is [0, num_zones]
        // For guard cells (f < 0 or f > num_zones), extrapolate linearly
        if (f < 0) {
            // Extrapolate below domain using first element's spacing
            double dr0 = (edges[1] - edges[0]) / Nz;
            return edges[0] + f * dr0;
        } else if (f > static_cast<int>(grid.num_zones)) {
            // Extrapolate above domain using last element's spacing
            double drN = (edges[Ne] - edges[Ne - 1]) / Nz;
            return edges[Ne] + (f - static_cast<int>(grid.num_zones)) * drN;
        } else if (f == static_cast<int>(grid.num_zones)) {
            return edges[Ne];
        } else {
            int e = f / Nz;
            int j = f % Nz;
            return edges[e] + (double(j) / Nz) * (edges[e + 1] - edges[e]);
        }
    }

    auto face_velocity(int f) const -> double {
        // Clamp to valid element range for velocity interpolation
        if (f < 0) {
            return v_edge[0];  // Use inner boundary velocity
        } else if (f >= static_cast<int>(grid.num_zones)) {
            return v_edge[Ne];  // Use outer boundary velocity
        } else {
            int e = f / Nz;
            double alpha = double(f % Nz) / Nz;
            return (1.0 - alpha) * v_edge[e] + alpha * v_edge[e + 1];
        }
    }

    auto face_area(int f) const -> double {
        if (grid.geom == geometry::planar) {
            return 1.0;
        } else {
            auto r = face_radius(f);
            return four_pi * r * r;
        }
    }

    auto cell_volume(int i) const -> double {
        if (grid.geom == geometry::planar) {
            return face_radius(i + 1) - face_radius(i);
        } else {
            auto rl = face_radius(i);
            auto rr = face_radius(i + 1);
            return four_pi / 3.0 * (rr * rr * rr - rl * rl * rl);
        }
    }

    auto cell_radius(int i) const -> double {
        return 0.5 * (face_radius(i) + face_radius(i + 1));
    }
};

// =============================================================================
// Pipeline stages
// =============================================================================

struct initial_state_t {
    static constexpr const char* name = "initial_state";
    grid_t grid;
    initial_condition ic;
    double tstart;
    unsigned num_elements;

    auto value(patch_t p) const -> patch_t {
        p.grid = grid;

        // Initialize element structure
        p.Ne = num_elements;
        p.Nz = grid.num_zones / num_elements;
        p.edges.resize(p.Ne + 1);
        p.v_edge.resize(p.Ne + 1, 0.0);
        p.edge_type.resize(p.Ne + 1, 0);
        p.fhat_edge.resize(p.Ne + 1, cons_t{});

        // Initialize edges uniformly across domain
        double r0 = grid.r0_initial;
        double r1 = grid.r1_initial;
        for (unsigned k = 0; k <= p.Ne; ++k) {
            p.edges[k] = r0 + k * (r1 - r0) / p.Ne;
        }

        // Initialize conserved variables
        for_each(p.interior, [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto rc = p.cell_radius(i);
            auto dv = p.cell_volume(i);
            auto prim = initial_primitive(ic, rc, tstart);
            auto cons = prim_to_cons(prim);
            p.cons[i] = cons * dv;
        });
        return p;
    }
};

struct local_dt_t {
    static constexpr const char* name = "compute_local_dt";
    double cfl;
    grid_t grid;
    double plm_theta;

    auto value(patch_t p) const -> patch_t {
        p.grid = grid;
        p.plm_theta = plm_theta;

        auto wavespeeds = lazy(p.interior, [&p](ivec_t<1> i) {
            return max_wavespeed(p.prim(i));
        });

        // Compute effective dr from minimum element width
        double dr_eff = std::numeric_limits<double>::max();
        for (unsigned k = 0; k < p.Ne; ++k) {
            dr_eff = min2(dr_eff, (p.edges[k + 1] - p.edges[k]) / p.Nz);
        }

        // Account for mesh motion: max edge velocity magnitude
        double max_vface = 0.0;
        for (unsigned k = 0; k <= p.Ne; ++k) {
            max_vface = max2(max_vface, std::fabs(p.v_edge[k]));
        }

        p.dt = cfl * dr_eff / (max(wavespeeds) + max_vface);
        return p;
    }
};

struct cache_rk_t {
    static constexpr const char* name = "cache_rk";
    auto value(patch_t p) const -> patch_t {
        p.time_rk = p.time;
        copy(p.cons_rk, p.cons);
        return p;
    }
};

struct cons_to_prim_t {
    static constexpr const char* name = "cons_to_prim";
    auto value(patch_t p) const -> patch_t {
        for_each(space(p.prim), [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto dv = p.cell_volume(i);
            p.prim[i] = cons_to_prim(p.cons[i] / dv);
        });
        return p;
    }
};

struct minimum_dt_t {
    double dt_max;
    static constexpr const char* name = "global_dt";
    using value_type = double;

    static auto init() -> double {
        return std::numeric_limits<double>::max();
    }

    auto reduce(double acc, const patch_t& p) const -> double {
        return std::min(acc, p.dt);
    }

    void finalize(double dt, patch_t& p) const {
        p.dt = std::min(dt, dt_max);
    }
};

struct exchange_cons_guard_t {
    static constexpr const char* name = "exchange_cons_guard";
    using space_t = index_space_t<1>;
    using buffer_t = array_view_t<cons_t, 1>;

    auto provides(const patch_t& p) const -> space_t {
        return p.interior;
    }

    void need(patch_t& p, auto request) const {
        auto lo = start(p.interior);
        auto hi = upper(p.interior);
        auto l_guard = index_space(lo - ivec(2), uvec(2));
        auto r_guard = index_space(hi, uvec(2));
        request(p.cons[l_guard]);
        request(p.cons[r_guard]);
    }

    auto data(const patch_t& p) const -> array_view_t<const cons_t, 1> {
        return p.cons[p.interior];
    }
};

struct apply_cons_boundary_conditions_t {
    static constexpr const char* name = "apply_bc";
    boundary_condition bc_lo;
    boundary_condition bc_hi;
    initial_condition ic;
    unsigned num_zones;  // global domain size
    double tstart;

    auto value(patch_t p) const -> patch_t {
        auto i0 = start(p.interior)[0];
        auto i1 = upper(p.interior)[0] - 1;
        auto& grid = p.grid;
        auto t = p.time;

        // Left boundary (patch starts at global origin)
        if (i0 == 0) {
            switch (bc_lo) {
                case boundary_condition::outflow:
                    for (int g = 0; g < 2; ++g) {
                        auto prim = cons_to_prim(p.cons[i0] / grid.cell_volume(i0, t));
                        p.cons[i0 - 1 - g] = prim_to_cons(prim) * grid.cell_volume(i0 - 1 - g, t);
                    }
                    break;
                case boundary_condition::inflow:
                    for (int g = 0; g < 2; ++g) {
                        auto i = i0 - 1 - g;
                        auto r = grid.cell_radius(i, t);
                        auto prim = initial_primitive(ic, r, tstart);
                        p.cons[i] = prim_to_cons(prim) * grid.cell_volume(i, t);
                    }
                    break;
                case boundary_condition::reflecting:
                    for (int g = 0; g < 2; ++g) {
                        auto prim = cons_to_prim(p.cons[i0 + g] / grid.cell_volume(i0 + g, t));
                        prim[1] = -prim[1];  // reflect radial velocity
                        p.cons[i0 - 1 - g] = prim_to_cons(prim) * grid.cell_volume(i0 - 1 - g, t);
                    }
                    break;
            }
        }

        // Right boundary (patch ends at global extent)
        if (static_cast<unsigned>(i1) == num_zones - 1) {
            switch (bc_hi) {
                case boundary_condition::outflow:
                    for (int g = 0; g < 2; ++g) {
                        auto prim = cons_to_prim(p.cons[i1] / grid.cell_volume(i1, t));
                        p.cons[i1 + 1 + g] = prim_to_cons(prim) * grid.cell_volume(i1 + 1 + g, t);
                    }
                    break;
                case boundary_condition::inflow:
                    for (int g = 0; g < 2; ++g) {
                        auto i = i1 + 1 + g;
                        auto r = grid.cell_radius(i, t);
                        auto prim = initial_primitive(ic, r, tstart);
                        p.cons[i] = prim_to_cons(prim) * grid.cell_volume(i, t);
                    }
                    break;
                case boundary_condition::reflecting:
                    for (int g = 0; g < 2; ++g) {
                        auto prim = cons_to_prim(p.cons[i1 - g] / grid.cell_volume(i1 - g, t));
                        prim[1] = -prim[1];  // reflect radial velocity
                        p.cons[i1 + 1 + g] = prim_to_cons(prim) * grid.cell_volume(i1 + 1 + g, t);
                    }
                    break;
            }
        }
        return p;
    }
};

struct compute_gradients_t {
    static constexpr const char* name = "compute_gradients";
    auto value(patch_t p) const -> patch_t {
        auto plm_theta = p.plm_theta;
        for_each(space(p.grad), [&](ivec_t<1> idx) {
            auto i = idx[0];
            p.grad[i] = plm_gradient(
                p.prim[i - 1],
                p.prim[i + 0],
                p.prim[i + 1],
                plm_theta
            );
        });
        return p;
    }
};

struct classify_element_boundaries_t {
    static constexpr const char* name = "classify_boundaries";
    double tol_rho;  // tolerance for density continuity check
    double tol_u;    // tolerance for four-velocity continuity check

    auto value(patch_t p) const -> patch_t {
        for (unsigned k = 0; k <= p.Ne; ++k) {
            int f = k * p.Nz;  // global face index at element boundary

            // Get L/R states at boundary (using guard cells for domain boundaries)
            auto pL = p.prim[f - 1];
            auto pR = p.prim[f];
            auto uL = prim_to_cons(pL);
            auto uR = prim_to_cons(pR);
            auto fL = prim_and_cons_to_flux(pL, uL);
            auto fR = prim_and_cons_to_flux(pR, uR);

            double v = 0.0;

            // New spec classification algorithm:
            // Step 1: Check if density is discontinuous
            if (!has_density_jump(pL, pR, tol_rho)) {
                // Density is continuous → neither shock nor contact
                p.edge_type[k] = 0;
                v = 0.5 * (beta(pL) + beta(pR));
            }
            // Step 2: Check if four-velocity is discontinuous (given density is discontinuous)
            else if (has_velocity_jump(pL, pR, tol_u)) {
                // Four-velocity is discontinuous → shock
                p.edge_type[k] = 2;
                v = compute_shock_velocity(pL, pR);
            }
            else {
                // Four-velocity is continuous → contact
                p.edge_type[k] = 1;
                v = 0.5 * (beta(pL) + beta(pR));
            }

            p.v_edge[k] = v;

            // Compute flux at discontinuity using F - v*U
            // Use interior-biased gradient (no PLM reconstruction at boundary)
            p.fhat_edge[k] = fL - uL * v;
        }
        return p;
    }
};

struct compute_fluxes_t {
    static constexpr const char* name = "compute_fluxes";
    riemann_solver solver = riemann_solver::hllc;

    auto value(patch_t p) const -> patch_t {
        for_each(space(p.fhat), [&](ivec_t<1> idx) {
            int f = idx[0];
            unsigned k = f / p.Nz;  // element boundary index

            // Check if this face is an element boundary with contact/shock treatment
            bool is_element_boundary = (f % static_cast<int>(p.Nz) == 0);
            bool has_special_flux = is_element_boundary && k <= p.Ne && p.edge_type[k] != 0;

            if (has_special_flux) {
                // Use pre-computed R-H flux at element boundary
                auto da = p.face_area(f);
                p.fhat[f] = p.fhat_edge[k] * da;
            } else {
                // Normal Riemann solve with PLM reconstruction
                auto da = p.face_area(f);
                auto vf = p.face_velocity(f);
                auto pl = p.prim[f - 1] + p.grad[f - 1] * 0.5;
                auto pr = p.prim[f + 0] - p.grad[f + 0] * 0.5;
                auto ul = prim_to_cons(pl);
                auto ur = prim_to_cons(pr);

                switch (solver) {
                    case riemann_solver::hlle:
                        p.fhat[f] = riemann_hlle(pl, pr, ul, ur, vf) * da;
                        break;
                    case riemann_solver::hllc:
                        p.fhat[f] = riemann_hllc(pl, pr, ul, ur, vf) * da;
                        break;
                    case riemann_solver::two_shock:
                        throw std::runtime_error("two_shock riemann solver not implemented for flux calculation");
                    case riemann_solver::exact:
                        throw std::runtime_error("exact riemann solver not implemented");
                }
            }
        });
        return p;
    }
};

struct update_conserved_t {
    static constexpr const char* name = "update_conserved";
    auto value(patch_t p) const -> patch_t {
        auto dt = p.dt;

        for_each(p.interior, [&](ivec_t<1> idx) {
            auto i = idx[0];
            p.cons[i] -= (p.fhat[i + 1] - p.fhat[i]) * dt;

            // Apply geometric source terms only for spherical geometry
            if (p.grid.geom == geometry::spherical) {
                auto rl = p.face_radius(i);
                auto rr = p.face_radius(i + 1);
                auto source = spherical_geometry_source_terms(p.prim[i], rl, rr);
                p.cons[i] += source * dt;
            }
        });

        // Move element boundaries
        for (unsigned k = 0; k <= p.Ne; ++k) {
            p.edges[k] += dt * p.v_edge[k];
        }

        // Enforce monotonicity
        constexpr double min_width = 1e-12;
        for (unsigned k = 0; k < p.Ne; ++k) {
            if (p.edges[k + 1] <= p.edges[k] + min_width) {
                p.edges[k + 1] = p.edges[k] + min_width;
            }
        }

        p.time += p.dt;
        return p;
    }
};

struct rk_average_t {
    static constexpr const char* name = "rk_average";
    double alpha;  // state = (1-alpha) * cached + alpha * current

    auto value(patch_t p) const -> patch_t {
        for_each(space(p.cons), [&](ivec_t<1> idx) {
            auto i = idx[0];
            p.cons[i] = p.cons_rk[i] * (1.0 - alpha) + p.cons[i] * alpha;
        });
        p.time = p.time_rk * (1.0 - alpha) + p.time * alpha;
        return p;
    }
};

// =============================================================================
// Custom serialization for patch_t
// =============================================================================

template<ArchiveWriter A>
void serialize(A& ar, const patch_t& p) {
    ar.begin_group();
    auto interior = cache(map(p.cons[p.interior], std::identity{}), memory::host, exec::cpu);
    serialize(ar, "cons", interior);
    serialize(ar, "Ne", p.Ne);
    serialize(ar, "Nz", p.Nz);
    serialize(ar, "edges", p.edges);
    ar.end_group();
}

template<ArchiveReader A>
auto deserialize(A& ar, patch_t& p) -> bool {
    if (!ar.begin_group()) return false;
    auto interior = cached_t<cons_t, 1>{};
    deserialize(ar, "cons", interior);
    unsigned Ne = 1, Nz = 0;
    std::vector<double> edges;
    deserialize(ar, "Ne", Ne);
    deserialize(ar, "Nz", Nz);
    deserialize(ar, "edges", edges);
    ar.end_group();
    p = patch_t(space(interior));
    copy(p.cons[p.interior], interior);
    p.Ne = Ne;
    p.Nz = Nz;
    p.edges = std::move(edges);
    p.v_edge.resize(p.Ne + 1, 0.0);
    p.edge_type.resize(p.Ne + 1, 0);
    p.fhat_edge.resize(p.Ne + 1, cons_t{});
    return true;
}

// =============================================================================
// 1D Special Relativistic Hydrodynamics Physics Module
// =============================================================================

struct blast {

    struct config_t {
        int rk_order = 1;
        double cfl = 0.4;
        double plm_theta = 1.5;
        boundary_condition bc_lo = boundary_condition::outflow;
        boundary_condition bc_hi = boundary_condition::outflow;
        riemann_solver riemann = riemann_solver::hllc;
        double tol_rho = 0.1;   // density discontinuity tolerance
        double tol_u = 1e-3;    // four-velocity discontinuity tolerance

        auto fields() const {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("plm_theta", plm_theta),
                field("bc_lo", bc_lo),
                field("bc_hi", bc_hi),
                field("riemann", riemann),
                field("tol_rho", tol_rho),
                field("tol_u", tol_u)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("plm_theta", plm_theta),
                field("bc_lo", bc_lo),
                field("bc_hi", bc_hi),
                field("riemann", riemann),
                field("tol_rho", tol_rho),
                field("tol_u", tol_u)
            );
        }
    };

    struct initial_t {
        unsigned int num_zones = 400;
        unsigned int num_partitions = 1;
        double inner_radius = 0.0;
        double outer_radius = 1.0;
        double tstart = 0.0;
        initial_condition ic = initial_condition::uniform;
        geometry geom = geometry::spherical;

        auto fields() const {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("num_partitions", num_partitions),
                field("inner_radius", inner_radius),
                field("outer_radius", outer_radius),
                field("tstart", tstart),
                field("ic", ic),
                field("geom", geom)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("num_partitions", num_partitions),
                field("inner_radius", inner_radius),
                field("outer_radius", outer_radius),
                field("tstart", tstart),
                field("ic", ic),
                field("geom", geom)
            );
        }
    };

    struct state_t {
        std::vector<patch_t> patches;
        double time;

        auto fields() const {
            return std::make_tuple(
                field("patches", patches),
                field("time", time)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("patches", patches),
                field("time", time)
            );
        }
    };

    using product_t = std::vector<cached_t<double, 1>>;

    struct exec_context_t {
        const config_t& config;
        const initial_t& initial;
        mutable parallel::scheduler_t scheduler;
        mutable perf::profiler_t profiler;

        exec_context_t(const config_t& cfg, const initial_t& ini)
            : config(cfg), initial(ini) {}

        void set_num_threads(std::size_t n) {
            scheduler.set_num_threads(n);
        }

        template<typename S>
        void execute(std::vector<patch_t>& patches, S s) const {
            parallel::execute(s, patches, scheduler, profiler);
        }
    };
};

// =============================================================================
// Physics interface implementation
// =============================================================================

auto default_physics_config(std::type_identity<blast>) -> blast::config_t {
    return {.rk_order = 1, .cfl = 0.4, .plm_theta = 1.5, .bc_lo = boundary_condition::outflow, .bc_hi = boundary_condition::outflow, .riemann = riemann_solver::hllc, .tol_rho = 0.1, .tol_u = 1e-3};
}

auto default_initial_config(std::type_identity<blast>) -> blast::initial_t {
    return {.num_zones = 400, .num_partitions = 1, .inner_radius = 0.0, .outer_radius = 1.0, .tstart = 0.0, .ic = initial_condition::uniform, .geom = geometry::spherical};
}

auto initial_state(const blast::exec_context_t& ctx) -> blast::state_t {
    using std::views::iota;
    using std::views::transform;

    auto& ini = ctx.initial;
    auto np = static_cast<int>(ini.num_partitions);
    auto S = index_space(ivec(0), uvec(ini.num_zones));
    auto num_elements = num_elements_for_ic(ini.ic);
    auto grid = grid_t{
        ini.inner_radius,
        ini.outer_radius,
        0.0,  // v_inner computed dynamically
        0.0,  // v_outer computed dynamically
        ini.num_zones,
        ini.geom
    };

    auto patches = to_vector(iota(0, np) | transform([&](int p) {
        return patch_t(subspace(S, np, p, 0));
    }));

    ctx.execute(patches, initial_state_t{grid, ini.ic, ini.tstart, num_elements});

    return {std::move(patches), 0.0};
}

void advance(blast::state_t& state, const blast::exec_context_t& ctx, double dt_max) {
    auto& ini = ctx.initial;
    auto& cfg = ctx.config;
    auto grid = grid_t{
        ini.inner_radius,
        ini.outer_radius,
        0.0,  // v_inner computed dynamically from gas velocities
        0.0,  // v_outer computed dynamically from gas velocities
        ini.num_zones,
        ini.geom
    };

    auto new_step = parallel::pipeline(
        cons_to_prim_t{},
        local_dt_t{cfg.cfl, grid, cfg.plm_theta},
        minimum_dt_t{dt_max},
        cache_rk_t{}
    );

    auto euler_step = parallel::pipeline(
        exchange_cons_guard_t{},
        apply_cons_boundary_conditions_t{cfg.bc_lo, cfg.bc_hi, ini.ic, ini.num_zones, ini.tstart},
        cons_to_prim_t{},
        compute_gradients_t{},
        classify_element_boundaries_t{cfg.tol_rho, cfg.tol_u},
        compute_fluxes_t{cfg.riemann},
        update_conserved_t{}
    );

    ctx.execute(state.patches, new_step);

    switch (cfg.rk_order) {
        case 1:
            ctx.execute(state.patches, euler_step);
            break;
        case 2:
            ctx.execute(state.patches, euler_step);
            ctx.execute(state.patches, euler_step);
            ctx.execute(state.patches, rk_average_t{0.5});
            break;
        case 3:
            ctx.execute(state.patches, euler_step);
            ctx.execute(state.patches, euler_step);
            ctx.execute(state.patches, rk_average_t{0.25});
            ctx.execute(state.patches, euler_step);
            ctx.execute(state.patches, rk_average_t{2.0 / 3.0});
            break;
        default:
            throw std::runtime_error("rk_order must be 1, 2, or 3");
    }

    state.time = state.patches[0].time;
}

auto zone_count(const blast::state_t& state, const blast::exec_context_t& ctx) -> std::size_t {
    return ctx.initial.num_zones;
}

auto names_of_time(std::type_identity<blast>) -> std::vector<std::string> {
    return {"t"};
}

auto names_of_timeseries(std::type_identity<blast>) -> std::vector<std::string> {
    return {"time", "total_mass", "total_energy", "max_lorentz"};
}

auto names_of_products(std::type_identity<blast>) -> std::vector<std::string> {
    return {"density", "velocity", "pressure", "lorentz_factor", "cell_r"};
}

auto get_time(const blast::state_t& state, const std::string& name) -> double {
    if (name == "t") {
        return state.time;
    }
    throw std::runtime_error("unknown time variable: " + name);
}

auto get_timeseries(
    const blast::config_t& cfg,
    const blast::initial_t& ini,
    const blast::state_t& state,
    const std::string& name
) -> double {
    if (name == "time") {
        return state.time;
    }

    auto total_mass = 0.0;
    auto total_energy = 0.0;
    auto max_lorentz = 1.0;

    for (const auto& p : state.patches) {
        // cons stores volume-integrated quantities, so total_mass = sum of cons[0]
        auto mass = lazy(p.interior, [&p](ivec_t<1> i) { return p.cons[i[0]][0]; });
        auto energy = lazy(p.interior, [&p](ivec_t<1> i) { return p.cons[i[0]][2]; });
        auto lorentz = lazy(p.interior, [&p](ivec_t<1> idx) {
            auto i = idx[0];
            auto dv = p.grid.cell_volume(i, p.time);
            return lorentz_factor(cons_to_prim(p.cons[i] / dv));
        });
        total_mass += sum(mass);
        total_energy += sum(energy);
        max_lorentz = max2(max_lorentz, max(lorentz));
    }

    if (name == "total_mass") return total_mass;
    if (name == "total_energy") return total_energy;
    if (name == "max_lorentz") return max_lorentz;

    throw std::runtime_error("unknown timeseries column: " + name);
}

auto get_product(
    const blast::state_t& state,
    const std::string& name,
    const blast::exec_context_t& ctx
) -> blast::product_t {
    using std::views::transform;

    // Ensure prim is up-to-date
    for (const auto& p : state.patches) {
        for_each(p.interior, [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto dv = p.grid.cell_volume(i, p.time);
            p.prim[i] = cons_to_prim(p.cons[i] / dv);
        });
    }

    auto make_product = [&](auto f) {
        return to_vector(state.patches | transform([f](const auto& p) {
            return cache(lazy(p.interior, [&p, f](ivec_t<1> i) {
                return f(p, i[0]);
            }), memory::host, exec::cpu);
        }));
    };

    if (name == "density") {
        return make_product([](const auto& p, int i) { return p.prim[i][0]; });
    }
    if (name == "velocity") {
        return make_product([](const auto& p, int i) { return beta(p.prim[i]); });
    }
    if (name == "pressure") {
        return make_product([](const auto& p, int i) { return p.prim[i][2]; });
    }
    if (name == "lorentz_factor") {
        return make_product([](const auto& p, int i) { return lorentz_factor(p.prim[i]); });
    }
    if (name == "cell_r") {
        return make_product([](const auto& p, int i) { return p.grid.cell_radius(i, p.time); });
    }
    throw std::runtime_error("unknown product: " + name);
}

auto get_profiler_data(const blast::exec_context_t& ctx)
    -> std::map<std::string, perf::profile_entry_t>
{
    return ctx.profiler.data();
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, const char* argv[])
{
    auto use_socket = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--socket") == 0 || std::strcmp(argv[i], "-s") == 0) {
            use_socket = true;
        }
    }

    auto physics = mist::driver::make_physics<blast>();
    auto state = mist::driver::state_t{};
    auto engine = mist::driver::engine_t{state, *physics};

    if (use_socket) {
        auto session = mist::driver::socket_session_t{engine};
        session.run();
    } else {
        auto session = mist::driver::repl_session_t{engine};
        session.run();
    }
    return 0;
}
