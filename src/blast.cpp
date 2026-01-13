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
#include <iostream>
#include <limits>
#include <optional>
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
// Enums
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

enum class edge_type {
    generic,  // neither shock nor contact
    contact,  // contact discontinuity
    shock     // shock
};

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

enum class external_model {
    sod,
    blast_wave,
    wind,
    uniform,
    four_state,
    mignone_bodo,
};

auto to_string(external_model model) -> const char* {
    switch (model) {
        case external_model::sod: return "sod";
        case external_model::blast_wave: return "blast_wave";
        case external_model::wind: return "wind";
        case external_model::uniform: return "uniform";
        case external_model::four_state: return "four_state";
        case external_model::mignone_bodo: return "mignone_bodo";
    }
    return "unknown";
}

auto from_string(std::type_identity<external_model>, const std::string& s) -> external_model {
    if (s == "sod") return external_model::sod;
    if (s == "blast_wave") return external_model::blast_wave;
    if (s == "wind") return external_model::wind;
    if (s == "uniform") return external_model::uniform;
    if (s == "four_state") return external_model::four_state;
    if (s == "mignone_bodo") return external_model::mignone_bodo;
    throw std::runtime_error("unknown initial condition: " + s);
}

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
// Math utility functions
// =============================================================================

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

static constexpr double gamma_law_index = 4.0 / 3.0;
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
    return rho + pre * (1.0 + 1.0 / (gamma_law_index - 1.0));
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
    auto gm = gamma_law_index;
    auto m = cons[0];
    auto tau = cons[2];
    auto ss = momentum_squared(cons);
    auto w0 = 0.0;

    int n;
    for (n = 0; n < newton_iter_max; ++n) {
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
    if (n == newton_iter_max) {
        throw std::runtime_error("cons_to_prim: Newton iteration failed to converge");
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
    return gamma_law_index * pre / rho_h;
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

static auto max_wavespeed(prim_t p) -> double {
    auto ws = outer_wavespeeds(p);
    return max2(std::fabs(ws[0]), std::fabs(ws[1]));
}

static auto spherical_geometry_source_terms(prim_t p, double r0, double r1) -> cons_t {
    // Eqn A8 in Zhang & MacFadyen (2006), integrated over the spherical shell
    // between r0 and r1, and specializing to radial velocity only.
    // Source = 4pi p (r1^2 - r0^2) to match the area-integrated fluxes
    auto pg = p[2];
    auto dr2 = std::pow(r1, 2) - std::pow(r0, 2);
    auto srdot = four_pi * pg * dr2;
    return cons_t{0.0, srdot, 0.0};
}

// =============================================================================
// Discontinuity detection helpers
// =============================================================================

static auto reldiff(double a, double b) -> double {
    using std::fabs;
    return fabs(a - b) / (0.5 * (fabs(a) + fabs(b)) + 1e-14);
}

static auto reldiff(cons_t a, cons_t b) -> double {
    using std::fabs;
    auto diff = fabs(a[0] - b[0]) + fabs(a[1] - b[1]) + fabs(a[2] - b[2]);
    auto scale = 0.5 * (fabs(a[0]) + fabs(b[0]) + fabs(a[1]) + fabs(b[1]) + fabs(a[2]) + fabs(b[2])) + 1e-14;
    return diff / scale;
}

// Compute shock velocity using relativistic jump relations
static auto compute_shock_velocity(prim_t pL, prim_t pR) -> double {
    using std::sqrt;
    using std::pow;
    constexpr double gh = gamma_law_index;  // 4/3 (g-hat as in Blandford-McKee notation)

    double uu, ud;
    double dir;

    if (pL[2] > pR[2]) {
        // Left is downstream (shocked), right is upstream (unshocked)
        ud = pL[1];
        uu = pR[1];
        dir = +1.0;  // shock propagates left to right
    } else {
        // Right is downstream (shocked), left is upstream (unshocked)
        ud = pR[1];
        uu = pL[1];
        dir = -1.0;  // shock propagates right to left
    }

    // Relative Lorentz factor
    double gl = sqrt(1.0 + uu * uu) * sqrt(1.0 + ud * ud) - uu * ud;

    // Shock Lorentz factor (in upstream rest frame)
    double gs = sqrt((gl + 1.0) * pow(gh * (gl - 1.0) + 1.0, 2.0) / (gh * (2.0 - gh) * (gl - 1.0) + 2.0));

    // Shock velocity in upstream rest frame
    double vs = dir * sqrt(1.0 - 1.0 / (gs * gs));

    // Upstream velocity in lab frame
    double vu = uu / sqrt(1.0 + uu * uu);

    // Boost shock velocity to lab frame using relativistic velocity addition
    return (vu + vs) / (1.0 + vu * vs);
}

// Check if states satisfy shock jump conditions (Rankine-Hugoniot)
// Returns true if F_L - v_s * U_L ~ F_R - v_s * U_R
static auto satisfies_shock_jump(prim_t pL, prim_t pR, double shock_tol) -> bool {
    auto v_s = compute_shock_velocity(pL, pR);
    auto uL = prim_to_cons(pL);
    auto uR = prim_to_cons(pR);
    auto fL = prim_and_cons_to_flux(pL, uL);
    auto fR = prim_and_cons_to_flux(pR, uR);
    auto flux_L = fL - uL * v_s;
    auto flux_R = fR - uR * v_s;
    return reldiff(flux_L, flux_R) < shock_tol;
}

// Check if states satisfy contact discontinuity conditions
// Returns true if v_L ~ v_R and p_L ~ p_R
static auto satisfies_contact_jump(prim_t pL, prim_t pR, double contact_tol) -> bool {
    auto vL = beta(pL);
    auto vR = beta(pR);
    auto pL_pressure = pL[2];
    auto pR_pressure = pR[2];
    return reldiff(vL, vR) < contact_tol && reldiff(pL_pressure, pR_pressure) < contact_tol;
}

// =============================================================================
// Riemann solvers
// =============================================================================

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
            throw std::runtime_error("riemann_hllc: a* is NaN");
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

// =============================================================================
// Data structures
// =============================================================================

struct grid_t {
    double r0 = 0.0;              // left edge position
    double r1 = 0.0;              // right edge position
    double v0 = 0.0;              // left edge velocity
    double v1 = 0.0;              // right edge velocity
    index_space_t<1> space;       // index space for this grid
    geometry geom = geometry::spherical;

    auto num_zones() const -> unsigned {
        return shape(space)[0];
    }

    auto dr() const -> double {
        return (r1 - r0) / num_zones();
    }

    auto face_radius(int f) const -> double {
        auto i0 = start(space)[0];
        auto i1 = upper(space)[0];
        double alpha = double(f - i0) / double(i1 - i0);
        return (1.0 - alpha) * r0 + alpha * r1;
    }

    auto cell_radius(int i) const -> double {
        return 0.5 * (face_radius(i) + face_radius(i + 1));
    }

    auto face_velocity(int f) const -> double {
        auto i0 = start(space)[0];
        auto i1 = upper(space)[0];
        double alpha = double(f - i0) / double(i1 - i0);
        return (1.0 - alpha) * v0 + alpha * v1;
    }

    auto face_area(int f) const -> double {
        if (geom == geometry::planar) {
            return 1.0;
        } else {
            auto r = face_radius(f);
            return four_pi * r * r;
        }
    }

    auto cell_volume(int i) const -> double {
        if (geom == geometry::planar) {
            return face_radius(i + 1) - face_radius(i);
        } else {
            auto rl = face_radius(i);
            auto rr = face_radius(i + 1);
            return four_pi / 3.0 * (rr * rr * rr - rl * rl * rl);
        }
    }
};

struct initial_t {
    unsigned int num_zones = 400;
    unsigned int num_patches = 1;
    double inner_radius = 0.0;
    double outer_radius = 1.0;
    double tstart = 0.0;
    external_model model = external_model::uniform;
    geometry geom = geometry::spherical;
    double four_state_dl = 10.0;
    double four_state_ul = 10.0;
    double four_state_dr = 1.0;
    double four_state_ur = 0.0;
    double cold_temp = 1e-6;

    auto fields() const {
        return std::make_tuple(
            field("num_zones", num_zones),
            field("num_patches", num_patches),
            field("inner_radius", inner_radius),
            field("outer_radius", outer_radius),
            field("tstart", tstart),
            field("model", model),
            field("geom", geom),
            field("four_state_dl", four_state_dl),
            field("four_state_ul", four_state_ul),
            field("four_state_dr", four_state_dr),
            field("four_state_ur", four_state_ur),
            field("cold_temp", cold_temp)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("num_zones", num_zones),
            field("num_patches", num_patches),
            field("inner_radius", inner_radius),
            field("outer_radius", outer_radius),
            field("tstart", tstart),
            field("model", model),
            field("geom", geom),
            field("four_state_dl", four_state_dl),
            field("four_state_ul", four_state_ul),
            field("four_state_dr", four_state_dr),
            field("four_state_ur", four_state_ur),
            field("cold_temp", cold_temp)
        );
    }
};

// Returns the 3 discontinuity positions for four_state: [r_rs, r_cd, r_fs]
static auto four_state_discontinuities(const initial_t& ini) -> std::array<double, 3> {
    auto dl = ini.four_state_dl;
    auto ul = ini.four_state_ul;
    auto dr = ini.four_state_dr;
    auto ur = ini.four_state_ur;

    auto sol = riemann::solve_two_shock(dl, ul, dr, ur);
    auto [v_rs, v_cd, v_fs] = riemann::compute_discontinuity_velocities({dl, ul, dr, ur}, sol);

    double r_rs = 1.0 + v_rs * ini.tstart;
    double r_cd = 1.0 + v_cd * ini.tstart;
    double r_fs = 1.0 + v_fs * ini.tstart;

    return {r_rs, r_cd, r_fs};
}

// Returns patch edge positions based on initial condition
static auto initial_patch_edges(const initial_t& ini) -> std::vector<double> {
    auto np = ini.num_patches;
    auto edges = std::vector<double>(np + 1);

    switch (ini.model) {
        case external_model::four_state: {
            if (np < 2 || np % 2 != 0) {
                throw std::runtime_error("four_state requires even number of patches >= 2");
            }
            if (ini.tstart <= 0.0) {
                throw std::runtime_error("four_state requires tstart > 0");
            }
            auto [r_rs, r_cd, r_fs] = four_state_discontinuities(ini);
            auto half = np / 2;

            // Domain boundaries are the shocks, contact in the middle
            edges[0] = r_rs;
            edges[half] = r_cd;
            edges[np] = r_fs;

            // Uniform distribution between r_rs and r_cd (left half)
            for (unsigned i = 1; i < half; ++i) {
                double alpha = double(i) / half;
                edges[i] = r_rs + alpha * (r_cd - r_rs);
            }

            // Uniform distribution between r_cd and r_fs (right half)
            for (unsigned i = half + 1; i < np; ++i) {
                double alpha = double(i - half) / half;
                edges[i] = r_cd + alpha * (r_fs - r_cd);
            }
            break;
        }
        case external_model::sod:
        case external_model::blast_wave:
        case external_model::wind:
        case external_model::uniform:
        case external_model::mignone_bodo:
        default:
            // Uniform edge distribution
            for (unsigned i = 0; i <= np; ++i) {
                double alpha = double(i) / np;
                edges[i] = ini.inner_radius + alpha * (ini.outer_radius - ini.inner_radius);
            }
            break;
    }
    return edges;
}

static auto initial_hydrodynamics(const initial_t& ini, double r, double t) -> prim_t {
    switch (ini.model) {
        case external_model::sod:
            if (r < 1.0) {
                return prim_t{1.0, 0.0, 1.0};
            } else {
                return prim_t{0.125, 0.0, 0.1};
            }
        case external_model::blast_wave:
            if (r < 1.0) {
                return prim_t{1.0, 0.0, 1000.0};
            } else {
                return prim_t{1.0, 0.0, 0.01};
            }
        case external_model::wind: {
            auto rho = 1.0 / (r * r);
            auto u = 1.0;  // gamma-beta = 1.0
            auto pre = 1e-6 * rho;
            return prim_t{rho, u, pre};
        }
        case external_model::uniform:
            return prim_t{1.0, 0.0, 1.0};
        case external_model::four_state: {
            auto dl = ini.four_state_dl;
            auto ul = ini.four_state_ul;
            auto dr = ini.four_state_dr;
            auto ur = ini.four_state_ur;

            // Solve two-shock Riemann problem
            auto sol = riemann::solve_two_shock(dl, ul, dr, ur);

            // Compute discontinuity velocities
            auto [v_rs, v_cd, v_fs] = riemann::compute_discontinuity_velocities({dl, ul, dr, ur}, sol);

            // Positions at time t (shocks launched from r=1.0)
            double r_rs = 1.0 + v_rs * t;
            double r_cd = 1.0 + v_cd * t;
            double r_fs = 1.0 + v_fs * t;

            // Determine which region r falls into
            if (r < r_rs) {
                // Region 4: original left state
                return prim_t{dl, ul, ini.cold_temp * dl};
            } else if (r < r_cd) {
                // Region 3: shocked left material
                return prim_t{sol.d3, sol.u, sol.p};
            } else if (r < r_fs) {
                // Region 2: shocked right material
                return prim_t{sol.d2, sol.u, sol.p};
            } else {
                // Region 1: original right state
                return prim_t{dr, ur, ini.cold_temp * dr};
            }
        }
        case external_model::mignone_bodo:
            if (r < 5.0) {
                return prim_t{1.0, 2.065, 1.0};
            } else {
                return prim_t{1.0, 0.0, 10.0};
            }
    }
    assert(false);
}

// External hydrodynamics for boundary conditions only
// Returns the upstream (unshocked) state appropriate for each boundary
static auto external_hydrodynamics(
    const initial_t& ini,
    double r,
    geometry geom,
    bool is_left_boundary
) -> prim_t {
    switch (ini.model) {
        case external_model::four_state: {
            if (is_left_boundary) {
                // Region 4: unshocked ejecta with 1/r² density profile in spherical
                auto dl = ini.four_state_dl;
                auto ul = ini.four_state_ul;
                double rho = (geom == geometry::spherical) ? dl / (r * r) : dl;
                return prim_t{rho, ul, ini.cold_temp * rho};
            } else {
                // Region 1: unshocked ambient (uniform density)
                auto dr = ini.four_state_dr;
                auto ur = ini.four_state_ur;
                return prim_t{dr, ur, ini.cold_temp * dr};
            }
        }
        default:
            // For other models, fall back to initial state at t=0
            return initial_hydrodynamics(ini, r, 0.0);
    }
}

// Source of truth for serialization and RK averaging
// Note: RK averaging uses lerp (linear interpolation) formula: (1-a)*a + a*b
// Note: cons array contains only interior cells (no guard zones)
struct truth_state_t {
    cached_t<cons_t, 1> cons;  // conserved variables (interior only, no guard zones)
    double time = 0.0;
    double r0 = 0.0;           // position of domain interior left edge
    double r1 = 0.0;           // position of domain interior right edge

    truth_state_t() = default;

    truth_state_t(index_space_t<1> s)
        : cons(cache(fill(s, cons_t{0.0, 0.0, 0.0}), memory::host, exec::cpu))
        , time(0.0)
        , r0(0.0)
        , r1(0.0)
    {
    }
};

struct patch_t {
    index_space_t<1> space;    // index space for this patch
    geometry geom = geometry::spherical;

    truth_state_t truth;       // current state
    truth_state_t truth_rk;    // cached state for RK averaging

    mutable double dt = 0.0;
    mutable double plm_theta = 1.5;

    // Edge velocities and types (computed by classify_patch_edges)
    mutable double v0 = 0.0;
    mutable double v1 = 0.0;
    mutable edge_type e0 = edge_type::generic;
    mutable edge_type e1 = edge_type::generic;
    mutable std::optional<cons_t> discontinuity_flux_l;
    mutable std::optional<cons_t> discontinuity_flux_r;

    // Temporary buffers (mutable, recomputed each step)
    mutable cached_t<prim_t, 1> prim;
    mutable cached_t<prim_t, 1> grad;
    mutable cached_t<cons_t, 1> fhat;
    mutable bool prim_current;
    mutable bool grad_current;
    mutable bool fhat_current;

    patch_t() = default;

    patch_t(index_space_t<1> s)
        : space(s)
        , truth(s)
        , truth_rk(s)
        , prim(cache(fill(expand(s, 2), prim_t{0.0, 0.0, 0.0}), memory::host, exec::cpu))
        , grad(cache(fill(expand(s, 1), prim_t{0.0, 0.0, 0.0}), memory::host, exec::cpu))
        , fhat(cache(fill(index_space(start(s), shape(s) + uvec(1)), cons_t{0.0, 0.0, 0.0}), memory::host, exec::cpu))
        , prim_current(false)
        , grad_current(false)
        , fhat_current(false)
    {
    }

    void invalidate_mutable_buffers() const {
        prim_current = false;
        grad_current = false;
        fhat_current = false;
    }

    // Construct grid on demand from truth and stored edge state
    auto grid() const -> grid_t {
        return {truth.r0, truth.r1, v0, v1, space, geom};
    }
};

// =============================================================================
// Pipeline stages
// =============================================================================

struct initial_state_t {
    static constexpr const char* name = "initial_state";
    initial_t ini;

    auto value(patch_t p) const -> patch_t {
        // Compute this patch's edge positions from its index space
        auto i0 = start(p.space)[0];
        auto edges = initial_patch_edges(ini);
        auto np = ini.num_patches;
        unsigned zones_per_patch = ini.num_zones / np;
        unsigned patch_idx = i0 / zones_per_patch;

        p.truth.r0 = edges[patch_idx];
        p.truth.r1 = edges[patch_idx + 1];

        if (ini.model == external_model::four_state) {
            // Classify edge types for four_state
            auto half = np / 2;
            auto edge_type_of = [&](unsigned edge_idx) {
                if (edge_idx == 0 || edge_idx == np) return edge_type::shock;
                if (edge_idx == half) return edge_type::contact;
                return edge_type::generic;
            };
            p.e0 = edge_type_of(patch_idx);
            p.e1 = edge_type_of(patch_idx + 1);
        } else {
            p.e0 = edge_type::generic;
            p.e1 = edge_type::generic;
        }

        p.v0 = 0.0;
        p.v1 = 0.0;
        p.geom = ini.geom;
        p.truth.time = ini.tstart;

        // Initialize conserved variables
        for_each(p.space, [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto rc = p.grid().cell_radius(i);
            auto dv = p.grid().cell_volume(i);
            auto prim = initial_hydrodynamics(ini, rc, ini.tstart);
            auto cons = prim_to_cons(prim);
            p.truth.cons[i] = cons * dv;
        });
        return p;
    }
};

struct local_dt_t {
    static constexpr const char* name = "compute_local_dt";
    double cfl;
    double plm_theta;

    auto value(patch_t p) const -> patch_t {
        p.plm_theta = plm_theta;

        auto wavespeeds = lazy(p.space, [&p](ivec_t<1> i) {
            return max_wavespeed(p.prim(i));
        });

        // Use grid dr for CFL
        double dr_eff = p.grid().dr();

        // Account for mesh motion: max edge velocity magnitude
        double max_vface = max2(std::fabs(p.v0), std::fabs(p.v1));

        p.dt = cfl * dr_eff / (max(wavespeeds) + max_vface);
        return p;
    }
};

struct cache_rk_t {
    static constexpr const char* name = "cache_rk";
    auto value(patch_t p) const -> patch_t {
        p.truth_rk.time = p.truth.time;
        p.truth_rk.r0 = p.truth.r0;
        p.truth_rk.r1 = p.truth.r1;
        copy(p.truth_rk.cons, p.truth.cons);
        return p;
    }
};

struct cons_to_prim_t {
    static constexpr const char* name = "cons_to_prim";
    auto value(patch_t p) const -> patch_t {
        if (p.prim_current) {
            return p;
        }
        // Only convert interior cells; guard zones filled by exchange/BC stages
        for_each(p.space, [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto dv = p.grid().cell_volume(i);
            p.prim[i] = cons_to_prim(p.truth.cons[i] / dv);
        });
        p.prim_current = true;
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

struct exchange_prim_guard_t {
    static constexpr const char* name = "exchange_prim_guard";
    using space_t = index_space_t<1>;
    using buffer_t = array_view_t<prim_t, 1>;

    auto provides(const patch_t& p) const -> space_t {
        return p.space;
    }

    void need(patch_t& p, auto request) const {
        auto lo = start(p.space);
        auto hi = upper(p.space);
        auto l_guard = index_space(lo - ivec(2), uvec(2));
        auto r_guard = index_space(hi, uvec(2));
        request(p.prim[l_guard]);
        request(p.prim[r_guard]);
    }

    auto data(const patch_t& p) const -> array_view_t<const prim_t, 1> {
        return p.prim[p.space];
    }
};

struct apply_prim_boundary_conditions_t {
    static constexpr const char* name = "apply_bc";
    boundary_condition bc_lo;
    boundary_condition bc_hi;
    initial_t ini;

    auto value(patch_t p) const -> patch_t {
        auto i0 = start(p.space)[0];
        auto i1 = upper(p.space)[0] - 1;

        // Left boundary (patch starts at global origin)
        if (i0 == 0) {
            switch (bc_lo) {
                case boundary_condition::outflow:
                    for (int g = 0; g < 2; ++g) {
                        p.prim[i0 - 1 - g] = p.prim[i0];
                    }
                    break;
                case boundary_condition::inflow:
                    for (int g = 0; g < 2; ++g) {
                        auto i = i0 - 1 - g;
                        auto r = p.grid().cell_radius(i);
                        p.prim[i] = external_hydrodynamics(ini, r, p.geom, true);
                    }
                    break;
                case boundary_condition::reflecting:
                    for (int g = 0; g < 2; ++g) {
                        p.prim[i0 - 1 - g] = p.prim[i0 + g];
                        p.prim[i0 - 1 - g][1] = -p.prim[i0 - 1 - g][1];  // reflect radial velocity
                    }
                    break;
            }
        }

        // Right boundary (patch ends at global extent)
        if (static_cast<unsigned>(i1) == ini.num_zones - 1) {
            switch (bc_hi) {
                case boundary_condition::outflow:
                    for (int g = 0; g < 2; ++g) {
                        p.prim[i1 + 1 + g] = p.prim[i1];
                    }
                    break;
                case boundary_condition::inflow:
                    for (int g = 0; g < 2; ++g) {
                        auto i = i1 + 1 + g;
                        auto r = p.grid().cell_radius(i);
                        p.prim[i] = external_hydrodynamics(ini, r, p.geom, false);
                    }
                    break;
                case boundary_condition::reflecting:
                    for (int g = 0; g < 2; ++g) {
                        p.prim[i1 + 1 + g] = p.prim[i1 - g];
                        p.prim[i1 + 1 + g][1] = -p.prim[i1 + 1 + g][1];  // reflect radial velocity
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
        if (p.grad_current) {
            return p;
        }
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
        p.grad_current = true;
        return p;
    }
};

struct classify_patch_edges_t {
    static constexpr const char* name = "classify_edges";
    double shock_tol;    // tolerance for shock jump condition check
    double contact_tol;  // tolerance for contact jump condition check

    // Classify a single edge and return (type, velocity)
    auto classify_edge(prim_t pL, prim_t pR) const -> std::pair<edge_type, double> {
        // Step 1: Check if states satisfy shock jump conditions
        if (satisfies_shock_jump(pL, pR, shock_tol)) {
            return {edge_type::shock, compute_shock_velocity(pL, pR)};
        }
        // Step 2: Check if states satisfy contact jump conditions
        if (satisfies_contact_jump(pL, pR, contact_tol)) {
            return {edge_type::contact, 0.5 * (beta(pL) + beta(pR))};
        }
        // Step 3: Otherwise generic
        return {edge_type::generic, 0.5 * (beta(pL) + beta(pR))};
    }

    auto value(patch_t p) const -> patch_t {
        auto i0 = start(p.space)[0];
        auto i1 = upper(p.space)[0];

        // Reset discontinuity fluxes
        p.discontinuity_flux_l.reset();
        p.discontinuity_flux_r.reset();

        // Classify left edge (face i0)
        {
            auto pL = p.prim[i0 - 1];
            auto pR = p.prim[i0];
            auto [et, vel] = classify_edge(pL, pR);
            p.e0 = et;
            p.v0 = vel;

            // Compute flux at discontinuity using interior state (pR)
            if (et != edge_type::generic) {
                auto uR = prim_to_cons(pR);
                auto fR = prim_and_cons_to_flux(pR, uR);
                p.discontinuity_flux_l = fR - uR * vel;
            }
        }

        // Classify right edge (face i1)
        {
            auto pL = p.prim[i1 - 1];
            auto pR = p.prim[i1];
            auto [et, vel] = classify_edge(pL, pR);
            p.e1 = et;
            p.v1 = vel;

            // Compute flux at discontinuity using interior state (pL)
            if (et != edge_type::generic) {
                auto uL = prim_to_cons(pL);
                auto fL = prim_and_cons_to_flux(pL, uL);
                p.discontinuity_flux_r = fL - uL * vel;
            }
        }

        return p;
    }
};

struct compute_fluxes_t {
    static constexpr const char* name = "compute_fluxes";
    riemann_solver solver = riemann_solver::hllc;

    auto value(patch_t p) const -> patch_t {
        if (p.fhat_current) {
            return p;
        }
        auto i0 = start(p.space)[0];
        auto i1 = upper(p.space)[0];

        for_each(space(p.fhat), [&](ivec_t<1> idx) {
            int f = idx[0];
            auto da = p.grid().face_area(f);

            // Override flux at left boundary if it's a discontinuity
            if (f == i0 && p.discontinuity_flux_l) {
                p.fhat[f] = *p.discontinuity_flux_l * da;
                return;
            }

            // Override flux at right boundary if it's a discontinuity
            if (f == i1 && p.discontinuity_flux_r) {
                p.fhat[f] = *p.discontinuity_flux_r * da;
                return;
            }

            // Riemann solve with PLM reconstruction
            auto vf = p.grid().face_velocity(f);
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
        });
        p.fhat_current = true;
        return p;
    }
};

struct update_conserved_t {
    static constexpr const char* name = "update_conserved";
    auto value(patch_t p) const -> patch_t {
        auto dt = p.dt;

        for_each(p.space, [&](ivec_t<1> idx) {
            auto i = idx[0];
            p.truth.cons[i] -= (p.fhat[i + 1] - p.fhat[i]) * dt;

            // Apply geometric source terms only for spherical geometry
            if (p.geom == geometry::spherical) {
                auto rl = p.grid().face_radius(i);
                auto rr = p.grid().face_radius(i + 1);
                auto source = spherical_geometry_source_terms(p.prim[i], rl, rr);
                p.truth.cons[i] += source * dt;
            }
        });

        // Move grid edges
        p.truth.r0 += p.v0 * dt;
        p.truth.r1 += p.v1 * dt;
        p.truth.time += dt;

        p.invalidate_mutable_buffers();
        return p;
    }
};

struct rk_average_t {
    static constexpr const char* name = "rk_average";
    double alpha;  // state = lerp(cached, current, alpha) = (1-alpha) * cached + alpha * current

    auto value(patch_t p) const -> patch_t {
        for_each(space(p.truth.cons), [&](ivec_t<1> idx) {
            auto i = idx[0];
            p.truth.cons[i] = p.truth_rk.cons[i] * (1.0 - alpha) + p.truth.cons[i] * alpha;
        });
        p.truth.time = p.truth_rk.time * (1.0 - alpha) + p.truth.time * alpha;
        p.truth.r0 = p.truth_rk.r0 * (1.0 - alpha) + p.truth.r0 * alpha;
        p.truth.r1 = p.truth_rk.r1 * (1.0 - alpha) + p.truth.r1 * alpha;
        p.invalidate_mutable_buffers();
        return p;
    }
};

// =============================================================================
// Serialization
// =============================================================================

template<ArchiveWriter A>
void serialize(A& ar, const patch_t& p) {
    ar.begin_group();
    serialize(ar, "cons", p.truth.cons);
    serialize(ar, "time", p.truth.time);
    serialize(ar, "r0", p.truth.r0);
    serialize(ar, "r1", p.truth.r1);
    ar.end_group();
}

template<ArchiveReader A>
auto deserialize(A& ar, patch_t& p) -> bool {
    if (!ar.begin_group()) return false;
    auto truth = truth_state_t{};
    deserialize(ar, "cons", truth.cons);
    deserialize(ar, "time", truth.time);
    deserialize(ar, "r0", truth.r0);
    deserialize(ar, "r1", truth.r1);
    ar.end_group();

    p = patch_t(space(truth.cons));
    p.truth = std::move(truth);
    return true;
}

// =============================================================================
// Physics module
// =============================================================================

struct blast {

    struct config_t {
        int rk_order = 1;
        double cfl = 0.4;
        double plm_theta = 1.5;
        boundary_condition bc_lo = boundary_condition::outflow;
        boundary_condition bc_hi = boundary_condition::outflow;
        riemann_solver riemann = riemann_solver::hllc;
        double shock_tol = 0.0;
        double contact_tol = 0.0;

        auto fields() const {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("plm_theta", plm_theta),
                field("bc_lo", bc_lo),
                field("bc_hi", bc_hi),
                field("riemann", riemann),
                field("shock_tol", shock_tol),
                field("contact_tol", contact_tol)
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
                field("shock_tol", shock_tol),
                field("contact_tol", contact_tol)
            );
        }
    };

    using initial_t = ::initial_t;

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
// Physics interface
// =============================================================================

auto default_physics_config(std::type_identity<blast>) -> blast::config_t {
    return {
        .rk_order = 1,
        .cfl = 0.4,
        .plm_theta = 1.5,
        .bc_lo = boundary_condition::outflow,
        .bc_hi = boundary_condition::outflow,
        .riemann = riemann_solver::hllc,
        .shock_tol = 0.1,
        .contact_tol = 0.1
    };
}

auto default_initial_config(std::type_identity<blast>) -> blast::initial_t {
    return {
        .num_zones = 400,
        .num_patches = 1,
        .inner_radius = 0.0,
        .outer_radius = 1.0,
        .tstart = 0.0,
        .model = external_model::uniform,
        .geom = geometry::spherical
    };
}

auto initial_state(const blast::exec_context_t& ctx) -> blast::state_t {
    using std::views::iota;
    using std::views::transform;

    auto& ini = ctx.initial;

    if (ini.model == external_model::four_state && (ini.num_patches < 2 || ini.num_patches % 2 != 0)) {
        throw std::runtime_error("four_state requires even number of patches >= 2");
    }

    auto np = static_cast<int>(ini.num_patches);
    auto S = index_space(ivec(0), uvec(ini.num_zones));

    auto patches = to_vector(iota(0, np) | transform([&](int p) {
        return patch_t(subspace(S, np, p, 0));
    }));

    ctx.execute(patches, initial_state_t{ini});

    return {std::move(patches), ini.tstart};
}

void advance(blast::state_t& state, const blast::exec_context_t& ctx, double dt_max) {
    auto& ini = ctx.initial;
    auto& cfg = ctx.config;

    auto new_step = parallel::pipeline(
        cons_to_prim_t{},
        local_dt_t{cfg.cfl, cfg.plm_theta},
        minimum_dt_t{dt_max},
        cache_rk_t{}
    );

    auto euler_step = parallel::pipeline(
        cons_to_prim_t{},
        exchange_prim_guard_t{},
        apply_prim_boundary_conditions_t{cfg.bc_lo, cfg.bc_hi, ini},
        compute_gradients_t{},
        classify_patch_edges_t{cfg.shock_tol, cfg.contact_tol},
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

    state.time = state.patches[0].truth.time;
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
    const blast::state_t& state,
    const std::string& name,
    const blast::exec_context_t& ctx
) -> double {
    if (name == "time") {
        return state.time;
    }

    auto total_mass = 0.0;
    auto total_energy = 0.0;
    auto max_lorentz = 1.0;

    for (const auto& p : state.patches) {
        // cons stores volume-integrated quantities, so total_mass = sum of cons[0]
        auto mass = lazy(p.space, [&p](ivec_t<1> i) { return p.truth.cons[i[0]][0]; });
        auto energy = lazy(p.space, [&p](ivec_t<1> i) { return p.truth.cons[i[0]][2]; });
        auto lorentz = lazy(p.space, [&p](ivec_t<1> idx) {
            auto i = idx[0];
            auto dv = p.grid().cell_volume(i);
            return lorentz_factor(cons_to_prim(p.truth.cons[i] / dv));
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
        if (!p.prim_current) {
            for_each(p.space, [&](ivec_t<1> idx) {
                auto i = idx[0];
                auto dv = p.grid().cell_volume(i);
                p.prim[i] = cons_to_prim(p.truth.cons[i] / dv);
            });
            p.prim_current = true;
        }
    }

    auto make_product = [&](auto f) {
        return to_vector(state.patches | transform([f](const auto& p) {
            return cache(lazy(p.space, [&p, f](ivec_t<1> i) {
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
        return make_product([](const auto& p, int i) { return p.grid().cell_radius(i); });
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
