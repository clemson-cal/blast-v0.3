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
#include "mist/driver.hpp"
#include "mist/ndarray.hpp"
#include "mist/parallel/pipeline.hpp"
#include "mist/serialize.hpp"

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
// Spherical geometry
// =============================================================================

struct grid_t {
    double r0 = 0.0;  // inner radius of domain
    double dr = 0.0;  // radial zone spacing

    auto face_radius(int i) const -> double {
        return r0 + i * dr;
    }

    auto cell_radius(int i) const -> double {
        return r0 + (i + 0.5) * dr;
    }

    auto face_area(int i) const -> double {
        auto r = face_radius(i);
        return four_pi * r * r;
    }

    auto cell_volume(int i) const -> double {
        auto rl = face_radius(i);
        auto rr = face_radius(i + 1);
        return four_pi / 3.0 * (rr * rr * rr - rl * rl * rl);
    }
};

// =============================================================================
// Initial conditions
// =============================================================================

enum class initial_condition {
    sod,
    blast_wave,
    wind,
    uniform
};

auto to_string(initial_condition ic) -> const char* {
    switch (ic) {
        case initial_condition::sod: return "sod";
        case initial_condition::blast_wave: return "blast_wave";
        case initial_condition::wind: return "wind";
        case initial_condition::uniform: return "uniform";
    }
    return "unknown";
}

auto from_string(std::type_identity<initial_condition>, const std::string& s) -> initial_condition {
    if (s == "sod") return initial_condition::sod;
    if (s == "blast_wave") return initial_condition::blast_wave;
    if (s == "wind") return initial_condition::wind;
    if (s == "uniform") return initial_condition::uniform;
    throw std::runtime_error("unknown initial condition: " + s);
}

static auto initial_primitive(initial_condition ic, double r) -> prim_t {
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
};

// =============================================================================
// Pipeline stages
// =============================================================================

struct initial_state_t {
    static constexpr const char* name = "initial_state";
    grid_t grid;
    initial_condition ic;

    auto value(patch_t p) const -> patch_t {
        p.grid = grid;
        for_each(p.interior, [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto rc = grid.cell_radius(i);
            auto dv = grid.cell_volume(i);
            auto prim = initial_primitive(ic, rc);
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
        p.dt = cfl * grid.dr / max(wavespeeds);
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
            auto dv = p.grid.cell_volume(i);
            p.prim[i] = cons_to_prim(p.cons[i] / dv);
        });
        return p;
    }
};

struct global_dt_t {
    static constexpr const char* name = "global_dt";
    using value_type = double;

    static auto init() -> double {
        return std::numeric_limits<double>::max();
    }

    auto reduce(double acc, const patch_t& p) const -> double {
        return std::min(acc, p.dt);
    }

    void finalize(double dt, patch_t& p) const {
        p.dt = dt;
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

    auto value(patch_t p) const -> patch_t {
        auto i0 = start(p.interior)[0];
        auto i1 = upper(p.interior)[0] - 1;
        auto& grid = p.grid;

        // Left boundary (patch starts at global origin)
        if (i0 == 0) {
            switch (bc_lo) {
                case boundary_condition::outflow:
                    for (int g = 0; g < 2; ++g) {
                        auto prim = cons_to_prim(p.cons[i0] / grid.cell_volume(i0));
                        p.cons[i0 - 1 - g] = prim_to_cons(prim) * grid.cell_volume(i0 - 1 - g);
                    }
                    break;
                case boundary_condition::inflow:
                    for (int g = 0; g < 2; ++g) {
                        auto i = i0 - 1 - g;
                        auto r = grid.cell_radius(i);
                        auto prim = initial_primitive(ic, r);
                        p.cons[i] = prim_to_cons(prim) * grid.cell_volume(i);
                    }
                    break;
                case boundary_condition::reflecting:
                    for (int g = 0; g < 2; ++g) {
                        auto prim = cons_to_prim(p.cons[i0 + g] / grid.cell_volume(i0 + g));
                        prim[1] = -prim[1];  // reflect radial velocity
                        p.cons[i0 - 1 - g] = prim_to_cons(prim) * grid.cell_volume(i0 - 1 - g);
                    }
                    break;
            }
        }

        // Right boundary (patch ends at global extent)
        if (static_cast<unsigned>(i1) == num_zones - 1) {
            switch (bc_hi) {
                case boundary_condition::outflow:
                    for (int g = 0; g < 2; ++g) {
                        auto prim = cons_to_prim(p.cons[i1] / grid.cell_volume(i1));
                        p.cons[i1 + 1 + g] = prim_to_cons(prim) * grid.cell_volume(i1 + 1 + g);
                    }
                    break;
                case boundary_condition::inflow:
                    for (int g = 0; g < 2; ++g) {
                        auto i = i1 + 1 + g;
                        auto r = grid.cell_radius(i);
                        auto prim = initial_primitive(ic, r);
                        p.cons[i] = prim_to_cons(prim) * grid.cell_volume(i);
                    }
                    break;
                case boundary_condition::reflecting:
                    for (int g = 0; g < 2; ++g) {
                        auto prim = cons_to_prim(p.cons[i1 - g] / grid.cell_volume(i1 - g));
                        prim[1] = -prim[1];  // reflect radial velocity
                        p.cons[i1 + 1 + g] = prim_to_cons(prim) * grid.cell_volume(i1 + 1 + g);
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

struct compute_fluxes_t {
    static constexpr const char* name = "compute_fluxes";
    auto value(patch_t p) const -> patch_t {
        for_each(space(p.fhat), [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto da = p.grid.face_area(i);
            auto pl = p.prim[i - 1] + p.grad[i - 1] * 0.5;
            auto pr = p.prim[i + 0] - p.grad[i + 0] * 0.5;
            p.fhat[i] = riemann_hlle(pl, pr, prim_to_cons(pl), prim_to_cons(pr)) * da;
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
            auto rl = p.grid.face_radius(i);
            auto rr = p.grid.face_radius(i + 1);
            auto source = spherical_geometry_source_terms(p.prim[i], rl, rr);
            p.cons[i] -= (p.fhat[i + 1] - p.fhat[i]) * dt;
            p.cons[i] += source * dt;
        });
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
    ar.end_group();
}

template<ArchiveReader A>
auto deserialize(A& ar, patch_t& p) -> bool {
    if (!ar.begin_group()) return false;
    auto interior = cached_t<cons_t, 1>{};
    deserialize(ar, "cons", interior);
    ar.end_group();
    p = patch_t(space(interior));
    copy(p.cons[p.interior], interior);
    return true;
}

// =============================================================================
// 1D Special Relativistic Hydrodynamics Physics Module
// =============================================================================

struct srhd {

    struct config_t {
        int rk_order = 1;
        double cfl = 0.4;
        double plm_theta = 1.5;
        boundary_condition bc_lo = boundary_condition::outflow;
        boundary_condition bc_hi = boundary_condition::outflow;

        auto fields() const {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("plm_theta", plm_theta),
                field("bc_lo", bc_lo),
                field("bc_hi", bc_hi)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("plm_theta", plm_theta),
                field("bc_lo", bc_lo),
                field("bc_hi", bc_hi)
            );
        }
    };

    struct initial_t {
        unsigned int num_zones = 400;
        unsigned int num_partitions = 1;
        double inner_radius = 0.0;
        double outer_radius = 1.0;
        initial_condition ic = initial_condition::uniform;

        auto fields() const {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("num_partitions", num_partitions),
                field("inner_radius", inner_radius),
                field("outer_radius", outer_radius),
                field("ic", ic)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("num_partitions", num_partitions),
                field("inner_radius", inner_radius),
                field("outer_radius", outer_radius),
                field("ic", ic)
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

auto default_physics_config(std::type_identity<srhd>) -> srhd::config_t {
    return {.rk_order = 1, .cfl = 0.4, .plm_theta = 1.5};
}

auto default_initial_config(std::type_identity<srhd>) -> srhd::initial_t {
    return {.num_zones = 400, .num_partitions = 1, .inner_radius = 0.0, .outer_radius = 1.0, .ic = initial_condition::uniform};
}

auto initial_state(const srhd::exec_context_t& ctx) -> srhd::state_t {
    using std::views::iota;
    using std::views::transform;

    auto& ini = ctx.initial;
    auto& cfg = ctx.config;
    auto np = static_cast<int>(ini.num_partitions);
    auto S = index_space(ivec(0), uvec(ini.num_zones));
    auto grid = grid_t{ini.inner_radius, (ini.outer_radius - ini.inner_radius) / ini.num_zones};

    auto patches = to_vector(iota(0, np) | transform([&](int p) {
        return patch_t(subspace(S, np, p, 0));
    }));

    ctx.execute(patches, initial_state_t{grid, ini.ic});

    return {std::move(patches), 0.0};
}

void advance(srhd::state_t& state, const srhd::exec_context_t& ctx) {
    auto& ini = ctx.initial;
    auto& cfg = ctx.config;
    auto grid = grid_t{ini.inner_radius, (ini.outer_radius - ini.inner_radius) / ini.num_zones};

    auto new_step = parallel::pipeline(
        cons_to_prim_t{},
        local_dt_t{cfg.cfl, grid, cfg.plm_theta},
        global_dt_t{},
        cache_rk_t{}
    );

    auto euler_step = parallel::pipeline(
        exchange_cons_guard_t{},
        apply_cons_boundary_conditions_t{cfg.bc_lo, cfg.bc_hi, ini.ic, ini.num_zones},
        cons_to_prim_t{},
        compute_gradients_t{},
        compute_fluxes_t{},
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

auto zone_count(const srhd::state_t& state, const srhd::exec_context_t& ctx) -> std::size_t {
    return ctx.initial.num_zones;
}

auto names_of_time(std::type_identity<srhd>) -> std::vector<std::string> {
    return {"t"};
}

auto names_of_timeseries(std::type_identity<srhd>) -> std::vector<std::string> {
    return {"time", "total_mass", "total_energy", "max_lorentz"};
}

auto names_of_products(std::type_identity<srhd>) -> std::vector<std::string> {
    return {"density", "velocity", "pressure", "lorentz_factor", "cell_r"};
}

auto get_time(const srhd::state_t& state, const std::string& name) -> double {
    if (name == "t") {
        return state.time;
    }
    throw std::runtime_error("unknown time variable: " + name);
}

auto get_timeseries(
    const srhd::config_t& cfg,
    const srhd::initial_t& ini,
    const srhd::state_t& state,
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
            auto dv = p.grid.cell_volume(i);
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
    const srhd::state_t& state,
    const std::string& name,
    const srhd::exec_context_t& ctx
) -> srhd::product_t {
    using std::views::transform;

    // Ensure prim is up-to-date
    for (const auto& p : state.patches) {
        for_each(p.interior, [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto dv = p.grid.cell_volume(i);
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
        return make_product([](const auto& p, int i) { return p.grid.cell_radius(i); });
    }
    throw std::runtime_error("unknown product: " + name);
}

auto get_profiler_data(const srhd::exec_context_t& ctx)
    -> std::map<std::string, perf::profile_entry_t>
{
    return ctx.profiler.data();
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    mist::program_t<srhd> prog;
    mist::run(prog);
    return 0;
}
