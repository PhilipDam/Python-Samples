# polynomial_solver.py
# ------------------------------------------------------------
# Polynomial solvers for degrees 1..5.
#
# Key fix (critical):
#   Cardano cubic solver must NOT take independent principal cube roots
#   u = cbrt(u3), v = cbrt(v3)
#   because their branches may not satisfy u*v = -p/3.
#
# Correct approach:
#   u = cbrt(u3)
#   v = -p/(3*u)    (fallback to cbrt(v3) if |u| is tiny)
#
# This fix is essential for Ferrari quartic correctness.
# ------------------------------------------------------------

from __future__ import annotations
import cmath
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

EPS = 1e-12


# -----------------------------
# Debug structures for quintic
# -----------------------------

@dataclass(frozen=True)
class IterationLogEntry:
    k: int
    method: str     # "Newton", "Bisection", "Done"
    x: float
    fx: float
    dfx: float
    lo: float
    hi: float
    width: float


@dataclass(frozen=True)
class QuinticDebugInfo:
    original_coeffs: List[float]   # [a5..a0]
    bracket_lo: float
    bracket_hi: float
    f_lo: float
    f_hi: float
    found_root: float
    final_lo: float
    final_hi: float
    deflated_quartic_coeffs: List[float]  # [a4..a0]
    deflation_remainder: float
    iterations: List[IterationLogEntry]


@dataclass(frozen=True)
class SolveOutput:
    roots: List[complex]
    quintic_debug: Optional[QuinticDebugInfo] = None


# -----------------------------
# Complex helpers
# -----------------------------

def cbrt(z: complex) -> complex:
    """
    Principal complex cube root using polar form.
    More stable than z ** (1/3) for complex values.
    """
    r = abs(z)
    theta = cmath.phase(z)
    return (r ** (1.0 / 3.0)) * complex(math.cos(theta / 3.0), math.sin(theta / 3.0))


# -----------------------------
# Core solvers 1..4
# -----------------------------

def solve_linear(a1: float, a0: float) -> List[complex]:
    """a1*x + a0 = 0"""
    if abs(a1) < EPS:
        return []
    return [-a0 / a1]


def solve_quadratic(a2: complex, a1: complex, a0: complex) -> List[complex]:
    """a2*x^2 + a1*x + a0 = 0 (complex-capable)"""
    if abs(a2) < EPS:
        return solve_linear(a1, a0)

    disc = a1 * a1 - 4 * a2 * a0
    s = cmath.sqrt(disc)
    return [(-a1 + s) / (2 * a2), (-a1 - s) / (2 * a2)]


def solve_cubic(a3: float, a2: float, a1: float, a0: float) -> List[complex]:
    """
    Cardano cubic solver (complex-safe).
    ✅ Includes the critical branch-consistency fix:
       compute u once, then v = -p/(3*u) to enforce u*v = -p/3.
    """
    if abs(a3) < EPS:
        return solve_quadratic(a2, a1, a0)

    # Normalize: x^3 + A x^2 + B x + C = 0
    A = a2 / a3
    B = a1 / a3
    C = a0 / a3

    # Depress: x = t - A/3 => t^3 + p t + q = 0
    shift = A / 3.0
    p = B - (A * A) / 3.0
    q = (2.0 * A * A * A) / 27.0 - (A * B) / 3.0 + C

    delta = (q / 2.0) ** 2 + (p / 3.0) ** 3
    sqrt_delta = cmath.sqrt(delta)

    u3 = -q / 2.0 + sqrt_delta
    v3 = -q / 2.0 - sqrt_delta

    u = cbrt(u3)

    # ✅ FIX: enforce u*v = -p/3
    if abs(u) < 1e-14:
        v = cbrt(v3)  # fallback for rare near-zero u
    else:
        v = -p / (3.0 * u)

    omega = complex(-0.5,  math.sqrt(3) / 2.0)
    omega2 = complex(-0.5, -math.sqrt(3) / 2.0)

    t1 = u + v
    t2 = omega * u + omega2 * v
    t3 = omega2 * u + omega * v

    return [t1 - shift, t2 - shift, t3 - shift]


def solve_quartic(a4: float, a3: float, a2: float, a1: float, a0: float) -> List[complex]:
    """
    Ferrari quartic solver (complex-safe).
    Relies on solve_cubic() for the resolvent cubic, so the cubic fix is essential.
    """
    if abs(a4) < EPS:
        return solve_cubic(a3, a2, a1, a0)

    # Normalize: x^4 + A x^3 + B x^2 + C x + D = 0
    A = a3 / a4
    B = a2 / a4
    C = a1 / a4
    D = a0 / a4

    # Depress: x = y - A/4 => y^4 + p y^2 + q y + r = 0
    shift = A / 4.0
    p = B - 3.0 * A * A / 8.0
    q = C + (A * A * A) / 8.0 - (A * B) / 2.0
    r = D - 3.0 * (A ** 4) / 256.0 + (A * A * B) / 16.0 - (A * C) / 4.0

    # Biquadratic shortcut: q ≈ 0
    if abs(q) < EPS:
        z_roots = solve_quadratic(1.0, p, r)  # z = y^2
        out: List[complex] = []
        for z in z_roots:
            y = cmath.sqrt(z)
            out.extend([y - shift, -y - shift])
        return out

    # Resolvent cubic:
    # m^3 - (p/2)m^2 - r m + (p r)/2 - (q^2)/8 = 0
    m_roots = solve_cubic(
        1.0,
        -(p / 2.0),
        -r,
        (p * r) / 2.0 - (q * q) / 8.0
    )

    # Choose m that gives robust alpha = sqrt(2m - p)
    best_m = m_roots[0]
    best_alpha = cmath.sqrt(2.0 * best_m - p)
    best_mag = abs(best_alpha)
    for m in m_roots[1:]:
        alpha = cmath.sqrt(2.0 * m - p)
        if abs(alpha) > best_mag + 1e-15:
            best_m, best_alpha, best_mag = m, alpha, abs(alpha)

    m = best_m
    alpha = best_alpha
    beta = (-q) / (2.0 * alpha)

    # Two quadratics in y:
    r1 = solve_quadratic(1.0, -alpha, m - beta)
    r2 = solve_quadratic(1.0,  alpha, m + beta)

    return [r1[0] - shift, r1[1] - shift, r2[0] - shift, r2[1] - shift]


# -----------------------------
# Quintic: bracket + hybrid + deflate + quartic
# -----------------------------

def eval_poly(coeffs_desc: List[float], x: float) -> float:
    """Horner evaluation for descending coefficients [a_n..a_0]."""
    acc = 0.0
    for a in coeffs_desc:
        acc = acc * x + a
    return acc


def eval_poly_and_deriv(coeffs_desc: List[float], x: float) -> Tuple[float, float]:
    """
    Extended Horner:
      b = P(x), c = P'(x)
    """
    b = coeffs_desc[0]
    c = 0.0
    for a in coeffs_desc[1:]:
        c = c * x + b
        b = b * x + a
    return b, c


def synthetic_divide_desc(coeffs_desc: List[float], root: float) -> Tuple[List[float], float]:
    """
    Divide P(x) by (x - root) using synthetic division.
    Input: [a_n..a_0]
    Output: (quotient [b_{n-1}..b_0], remainder)
    """
    b: List[float] = [coeffs_desc[0]]
    for i in range(1, len(coeffs_desc)):
        b.append(coeffs_desc[i] + root * b[i - 1])
    remainder = b[-1]
    quotient = b[:-1]
    return quotient, remainder


def bracket_one_root(coeffs_desc: List[float]) -> Optional[Tuple[float, float, float, float]]:
    """
    Find lo,hi such that f(lo)*f(hi) <= 0, expanding symmetrically.
    For odd degree, this should succeed quickly if leading coefficient != 0.
    """
    an = coeffs_desc[0]
    if abs(an) < EPS:
        return None

    max_other = max(abs(a) for a in coeffs_desc[1:]) if len(coeffs_desc) > 1 else 0.0
    R = 1.0 + max_other / abs(an)

    lo, hi = -R, R
    flo, fhi = eval_poly(coeffs_desc, lo), eval_poly(coeffs_desc, hi)

    max_expand = 40
    i = 0
    while flo * fhi > 0 and i < max_expand:
        R *= 2
        lo, hi = -R, R
        flo, fhi = eval_poly(coeffs_desc, lo), eval_poly(coeffs_desc, hi)
        i += 1

    if flo * fhi > 0:
        return None
    return lo, hi, flo, fhi


def newton_bisection_hybrid_with_log(
    coeffs_desc: List[float],
    lo: float, hi: float,
    flo: float, fhi: float,
    max_iter: int = 200,
    tol_f: float = 1e-12,
    tol_x: float = 1e-12
) -> Tuple[float, float, float, List[IterationLogEntry]]:
    """
    Hybrid root solver inside [lo, hi], returns:
      (root, final_lo, final_hi, iteration_log)
    """
    x = 0.5 * (lo + hi)
    log: List[IterationLogEntry] = []

    for k in range(max_iter):
        f, df = eval_poly_and_deriv(coeffs_desc, x)
        width = hi - lo

        if abs(f) < tol_f or abs(width) < tol_x:
            log.append(IterationLogEntry(k, "Done", x, f, df, lo, hi, width))
            return x, lo, hi, log

        method = "Bisection"
        x_candidate = 0.5 * (lo + hi)

        if abs(df) > 1e-14:
            xn = x - f / df
            if lo < xn < hi:
                method = "Newton"
                x_candidate = xn

        fc = eval_poly(coeffs_desc, x_candidate)
        _, dfc = eval_poly_and_deriv(coeffs_desc, x_candidate)

        # Update bracket
        if flo * fc <= 0:
            hi, fhi = x_candidate, fc
        else:
            lo, flo = x_candidate, fc

        log.append(IterationLogEntry(k, method, x_candidate, fc, dfc, lo, hi, hi - lo))
        x = x_candidate

    # Max-iter fallback: return best x
    f, df = eval_poly_and_deriv(coeffs_desc, x)
    log.append(IterationLogEntry(max_iter, "Done", x, f, df, lo, hi, hi - lo))
    return x, lo, hi, log


def solve_quintic_with_debug(coeffs_desc: List[float]) -> SolveOutput:
    """
    Solve degree-5:
      1) bracket one real root
      2) hybrid Newton/bisection to refine it (log stored)
      3) deflate to quartic
      4) solve quartic (Ferrari)
    """
    if len(coeffs_desc) != 6:
        return SolveOutput([])

    if abs(coeffs_desc[0]) < EPS:
        # Degenerate: treat as quartic
        r = solve_quartic(*coeffs_desc[1:])
        return SolveOutput(r)

    br = bracket_one_root(coeffs_desc)
    if br is None:
        return SolveOutput([])

    lo, hi, flo, fhi = br
    root, final_lo, final_hi, it_log = newton_bisection_hybrid_with_log(coeffs_desc, lo, hi, flo, fhi)

    qcoeffs, rem = synthetic_divide_desc(coeffs_desc, root)  # quartic coeffs [a4..a0]
    quartic_roots = solve_quartic(*qcoeffs)

    dbg = QuinticDebugInfo(
        original_coeffs=list(coeffs_desc),
        bracket_lo=lo, bracket_hi=hi,
        f_lo=flo, f_hi=fhi,
        found_root=root,
        final_lo=final_lo, final_hi=final_hi,
        deflated_quartic_coeffs=list(qcoeffs),
        deflation_remainder=rem,
        iterations=it_log
    )

    roots = [complex(root, 0.0)] + quartic_roots
    return SolveOutput(roots, dbg)


def solve(degree: int, coeffs_desc: List[float], with_debug: bool = False) -> SolveOutput:
    """
    Solve degree 1..5 for descending coefficients [a_n..a_0].
    Returns SolveOutput so quintic can include debug info.
    """
    # Trim leading zeros defensively
    coeffs = list(coeffs_desc)
    while len(coeffs) > 1 and abs(coeffs[0]) < EPS:
        coeffs.pop(0)

    eff_deg = len(coeffs) - 1

    if eff_deg == 1:
        return SolveOutput(solve_linear(coeffs[0], coeffs[1]))
    if eff_deg == 2:
        return SolveOutput(solve_quadratic(coeffs[0], coeffs[1], coeffs[2]))
    if eff_deg == 3:
        return SolveOutput(solve_cubic(coeffs[0], coeffs[1], coeffs[2], coeffs[3]))
    if eff_deg == 4:
        return SolveOutput(solve_quartic(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]))
    if eff_deg == 5:
        if with_debug:
            return solve_quintic_with_debug(coeffs)
        else:
            out = solve_quintic_with_debug(coeffs)
            return SolveOutput(out.roots, None)

    return SolveOutput([])



# main.py
from polynomial_solver import solve

def fmt(z: complex) -> str:
    r = 0.0 if abs(z.real) < 1e-12 else z.real
    i = 0.0 if abs(z.imag) < 1e-12 else z.imag
    if i == 0:
        return f"{r:.12g}"
    if r == 0:
        return f"{i:.12g}i"
    sign = "+" if i >= 0 else "-"
    return f"{r:.12g} {sign} {abs(i):.12g}i"

if __name__ == "__main__":
    # First example quartic:
    # x^4 - 3x^3 - 3x^2 + 61x - 156 = 0
    coeffs = [1, -3, -3, 61, -156]
    out = solve(4, coeffs)

    print("Roots:")
    for z in out.roots:
        print(" ", fmt(z))