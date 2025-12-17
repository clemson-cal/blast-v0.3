/**
================================================================================
Two-Shock Relativistic Riemann Solver

Solves the two-shock Riemann problem for relativistic hydrodynamics. Given two
cold (p=0) initial states, computes the intermediate state consisting of:
- A forward shock (FS) propagating into the right state
- A reverse shock (RS) propagating into the left state
- A contact discontinuity (CD) separating shocked regions

The solution yields a common four-velocity and pressure in the contact region,
with different densities on either side of the contact.

Input:  dl, ul (left state density and four-velocity)
        dr, ur (right state density and four-velocity)
Output: u, p (common four-velocity and pressure in contact region)
        d2 (shocked right-state material)
        d3 (shocked left-state material)
================================================================================
*/

#pragma once

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace riemann {

/**
 * Input state for the two-shock Riemann problem
 * Left (l) and right (r) states with cold initial pressure (p=0)
 */
struct two_shock_input_t {
    double dl;  // comoving density, left state
    double ul;  // four-velocity, left state
    double dr;  // comoving density, right state
    double ur;  // four-velocity, right state
};

/**
 * Solution to the two-shock Riemann problem
 */
struct two_shock_solution_t {
    double u;   // common four-velocity in contact region
    double p;   // common pressure in contact region
    double d2;  // density in region 2 (shocked right-state material)
    double d3;  // density in region 3 (shocked left-state material)
};

namespace detail {

/**
 * Brent's root-finding method
 * Finds x such that f(x) = 0 in the interval [a, b]
 * Requires f(a) and f(b) to have opposite signs
 */
template<typename F>
auto brent_root(F f, double a, double b, double tol = 1e-12, int max_iter = 100) -> double {
    double fa = f(a);
    double fb = f(b);

    if (fa * fb > 0) {
        throw std::runtime_error("brent_root: f(a) and f(b) must have opposite signs");
    }

    if (std::fabs(fa) < std::fabs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;
    double fc = fa;
    bool mflag = true;
    double s = 0.0;
    double d = 0.0;

    for (int i = 0; i < max_iter; ++i) {
        if (std::fabs(b - a) < tol) {
            return b;
        }

        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions for bisection
        double tmp = (3 * a + b) / 4;
        bool cond1 = !((s > tmp && s < b) || (s < tmp && s > b));
        bool cond2 = mflag && std::fabs(s - b) >= std::fabs(b - c) / 2;
        bool cond3 = !mflag && std::fabs(s - b) >= std::fabs(c - d) / 2;
        bool cond4 = mflag && std::fabs(b - c) < tol;
        bool cond5 = !mflag && std::fabs(c - d) < tol;

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            // Bisection
            s = (a + b) / 2;
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (std::fabs(fa) < std::fabs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }

        if (std::fabs(fb) < tol) {
            return b;
        }
    }

    return b;
}

/**
 * The equation f(u) = 0 whose root gives the common four-velocity
 * in the contact region between the two shocks.
 *
 * Region labeling (following standard convention):
 *   Region 1 = right state (dr, ur) - e.g., CBM
 *   Region 4 = left state (dl, ul) - e.g., ejecta
 *   Region 2 = shocked region 1 material
 *   Region 3 = shocked region 4 material
 */
inline auto u_equation(double d1, double g1, double u1, double d4, double g4, double u4) {
    return [=](double u) -> double {
        double g = std::sqrt(1.0 + u * u);
        double g2 = g * g;
        double g1_2 = g1 * g1;
        double g4_2 = g4 * g4;

        double term1 = -1.0
            + g * g1
            + (-1.0 + 8.0 * g * g1) * u * u1
            + g2 * (4.0 - 8.0 * g1_2)
            + 4.0 * g1_2;

        double term2_d1 = d1 * (
            -1.0
            + g * (4.0 * g + g1)
            + (-1.0 + 8.0 * g * g1) * u * u1
            + 4.0 * (1.0 - 2.0 * g2) * g1_2
        );

        double term2_d4 = d4 * (
            1.0
            - g * (4.0 * g + g4)
            + u * (u4 - 8.0 * g * g4 * u4)
            + 4.0 * (-1.0 + 2.0 * g2) * g4_2
        );

        return term1 * (term2_d1 + term2_d4);
    };
}

/**
 * Compute pressure in the contact region (c = 1 units)
 */
inline auto compute_pressure(double g, double u, double d1, double u1, double g1) -> double {
    double g2 = g * g;
    double g1_2 = g1 * g1;

    return (d1 * (
        1.0
        - g * (4.0 * g + g1)
        + u * (u1 - 8.0 * g * g1 * u1)
        + 4.0 * (-1.0 + 2.0 * g2) * g1_2
    )) / 3.0;
}

/**
 * Compute density in region 2 (shocked right-state material)
 */
inline auto compute_d2(double p, double g, double u, double d1, double g1, double u1) -> double {
    double g2 = g * g;
    double numerator = d1 * p * (4.0 * g * g1 * u + u1 - 4.0 * u1 * g2);
    double denominator = p * u + d1 * (g - g1) * (-(g1 * u) + g * u1);
    return numerator / denominator;
}

/**
 * Compute density in region 3 (shocked left-state material)
 */
inline auto compute_d3(double p, double g, double u, double d4, double g4, double u4) -> double {
    double g2 = g * g;
    double numerator = d4 * p * (4.0 * g * g4 * u + u4 - 4.0 * u4 * g2);
    double denominator = p * u + d4 * (g - g4) * (-(g4 * u) + g * u4);
    return numerator / denominator;
}

} // namespace detail

/**
 * Solve the two-shock Riemann problem
 *
 * @param input  The input state containing densities and four-velocities
 *               for left (l) and right (r) cold (p=0) states
 * @param u_min  Lower bound for four-velocity search (default: -100)
 * @param u_max  Upper bound for four-velocity search (default: 100)
 * @param tol    Tolerance for root finding (default: 1e-12)
 * @return       The solution containing common u, g, p and densities d2, d3
 */
inline auto solve_two_shock(
    const two_shock_input_t& input,
    double u_min = -100.0,
    double u_max = 100.0,
    double tol = 1e-12
) -> two_shock_solution_t {
    // Map input to standard region labeling:
    // Region 1 = right state, Region 4 = left state
    double d1 = input.dr;
    double u1 = input.ur;
    double g1 = std::sqrt(1.0 + u1 * u1);

    double d4 = input.dl;
    double u4 = input.ul;
    double g4 = std::sqrt(1.0 + u4 * u4);

    auto f = detail::u_equation(d1, g1, u1, d4, g4, u4);

    // Find the root using Brent's method
    double u = detail::brent_root(f, u_min, u_max, tol);
    double g = std::sqrt(1.0 + u * u);

    // Compute pressure (same in regions 2 and 3)
    double p = detail::compute_pressure(g, u, d1, u1, g1);

    // Compute densities in shocked regions
    double d2 = detail::compute_d2(p, g, u, d1, g1, u1);
    double d3 = detail::compute_d3(p, g, u, d4, g4, u4);

    return {u, p, d2, d3};
}

/**
 * Solve the two-shock Riemann problem (direct arguments)
 */
inline auto solve_two_shock(
    double dl, double ul,
    double dr, double ur,
    double u_min = -100.0,
    double u_max = 100.0,
    double tol = 1e-12
) -> two_shock_solution_t {
    return solve_two_shock({dl, ul, dr, ur}, u_min, u_max, tol);
}

/**
 * Try to solve the two-shock Riemann problem, returning nullopt on failure
 */
inline auto try_solve_two_shock(
    const two_shock_input_t& input,
    double u_min = -100.0,
    double u_max = 100.0,
    double tol = 1e-12
) -> std::optional<two_shock_solution_t> {
    try {
        return solve_two_shock(input, u_min, u_max, tol);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

/**
 * Compute discontinuity velocities given the solution
 *
 * @param input     The input state
 * @param solution  The two-shock solution
 * @return          Vector of (reverse shock velocity, contact velocity, forward shock velocity)
 *                  as coordinate velocities (beta = v/c)
 */
inline auto compute_discontinuity_velocities(
    const two_shock_input_t& input,
    const two_shock_solution_t& solution
) -> std::array<double, 3> {
    // Map input to standard region labeling
    double d1 = input.dr;
    double u1 = input.ur;
    double g1 = std::sqrt(1.0 + u1 * u1);
    double beta1 = u1 / g1;

    double d4 = input.dl;
    double u4 = input.ul;
    double g4 = std::sqrt(1.0 + u4 * u4);
    double beta4 = u4 / g4;

    // Contact velocity from common four-velocity
    double g = std::sqrt(1.0 + solution.u * solution.u);
    double v_cd = solution.u / g;

    // Shock velocity from mass flux conservation:
    // rho1 * gamma1 * (v1 - vs) = rho2 * gamma * (v - vs)
    // Solving for vs:
    double d1_g1 = d1 * g1;
    double d2_g = solution.d2 * g;
    double v_fs = (d2_g * v_cd - d1_g1 * beta1) / (d2_g - d1_g1);

    double d4_g4 = d4 * g4;
    double d3_g = solution.d3 * g;
    double v_rs = (d3_g * v_cd - d4_g4 * beta4) / (d3_g - d4_g4);

    return {v_rs, v_cd, v_fs};
}

} // namespace riemann
