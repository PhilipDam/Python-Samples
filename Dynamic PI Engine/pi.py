"""
pi_engine.py — A dynamically-scaling π engine (pure Python)

What this provides
- Compute π to *any requested number of decimal digits* (limited by CPU/RAM).
- Automatically scales:
    - internal Decimal precision (digits + guard digits)
    - number of Chudnovsky terms (≈ 14.18 digits per term)
- Uses fast integer arithmetic + binary splitting (no external libraries).
- Caches the most recently computed high-precision π so repeated calls are cheap.

Why you previously saw decimal.InvalidOperation
- Decimal.quantize() uses the *current* decimal context precision.
- If you quantize outside a high-precision localcontext(), the default precision (often 28)
  can be too small, raising InvalidOperation.
- This version always quantizes under a sufficiently-large localcontext().

Usage
    from pi_engine import PiEngine
    eng = PiEngine()
    print(eng.pi_str(50))       # 50 digits after decimal
    print(eng.pi_str(2000))     # 2000 digits after decimal
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_FLOOR
from typing import Tuple


# -----------------------------
# Chudnovsky constants
# -----------------------------
_A = 13591409
_B = 545140134
_C = 640320
_C3_OVER_24 = (_C**3) // 24  # 640320^3 / 24 is exact integer


def _bs(a: int, b: int) -> Tuple[int, int, int]:
    """
    Binary splitting for the Chudnovsky series.

    Returns (P, Q, T) as integers.

    This scheme is arranged so we avoid huge factorials directly and get fast convergence.

    Combine rule (for intervals [a,m) and [m,b)):
        P = P1 * P2
        Q = Q1 * Q2
        T = T1 * Q2 + T2 * P1
    """
    if b - a == 1:
        k = a

        if k == 0:
            P = 1
            Q = 1
        else:
            # Pk = (6k-5)(2k-1)(6k-1)
            P = (6 * k - 5) * (2 * k - 1) * (6 * k - 1)
            # Qk = k^3 * (C^3/24)
            Q = (k * k * k) * _C3_OVER_24

        # Tk = Pk * (A + Bk)
        T = P * (_A + _B * k)

        # Alternating sign
        if k & 1:
            T = -T

        return P, Q, T

    m = (a + b) // 2
    P1, Q1, T1 = _bs(a, m)
    P2, Q2, T2 = _bs(m, b)

    P = P1 * P2
    Q = Q1 * Q2
    T = T1 * Q2 + T2 * P1
    return P, Q, T


@dataclass
class PiEngine:
    """
    A π engine that dynamically scales precision and caches results.

    Public API:
      - pi_decimal(digits): Decimal π truncated to `digits` digits after the decimal point.
      - pi_str(digits): string with exactly `digits` digits after decimal.

    Notes on rounding:
      - We use ROUND_FLOOR in quantize, which for positive π is truncation.
      - If you prefer standard rounding, change rounding mode to ROUND_HALF_EVEN, etc.
    """
    guard_digits: int = 20

    # Cache the last computed high-precision π and the precision used to compute it.
    _cached_pi: Decimal | None = None
    _cached_prec: int = 0  # decimal context precision used when caching

    def _required_terms(self, digits: int) -> int:
        """
        Chudnovsky yields about 14.181647... correct digits per term.
        Add a small safety margin.
        """
        if digits <= 0:
            return 1
        return int(digits / 14.181647462725477) + 3

    def _cache_support_digits(self) -> int:
        """
        Very conservative estimate of how many digits after the decimal the cached π can safely serve.

        The cached Decimal is rounded to _cached_prec significant digits (not decimal places).
        A conservative safe bound is:
            safe_digits_after_decimal ≈ _cached_prec - 2

        Explanation:
        - π is ~3.1415..., so it has 1 digit before the decimal.
        - Significant digits = 1 (integer part) + digits_after_decimal.
        - So digits_after_decimal ≈ significant_digits - 1.
        - We subtract an extra 1 for safety.
        """
        if self._cached_pi is None or self._cached_prec <= 0:
            return 0
        return max(0, self._cached_prec - 2)

    def pi_decimal(self, digits: int) -> Decimal:
        """
        Return π as a Decimal truncated (floored) to `digits` digits after the decimal point.
        """
        if digits < 0:
            raise ValueError("digits must be >= 0")

        # If cache is present and has enough precision, reuse it.
        cache_ok_digits = self._cache_support_digits()
        if self._cached_pi is not None and cache_ok_digits >= digits:
            return self._quantize_safe(self._cached_pi, digits, min_prec=self._cached_prec)

        # Compute a fresh value with sufficient precision.
        terms = self._required_terms(digits)
        prec = digits + self.guard_digits

        with localcontext() as ctx:
            ctx.prec = prec

            # Binary splitting gives integer Q and T for the partial sum
            _, Q, T = _bs(0, terms)

            # pi = 426880 * sqrt(10005) * Q / T
            sqrt_10005 = Decimal(10005).sqrt()
            pi = (Decimal(426880) * sqrt_10005 * Decimal(Q)) / Decimal(T)

            # Cache the *high-precision* value (rounded to ctx.prec significant digits)
            self._cached_pi = +pi
            self._cached_prec = prec

            # Quantize under the same high-precision context (safe)
            return self._quantize(pi, digits)

    def pi_str(self, digits: int) -> str:
        """
        Return π as a string with exactly `digits` digits after the decimal point.
        """
        pi_dec = self.pi_decimal(digits)

        # Avoid scientific notation.
        s = format(pi_dec, "f")

        if digits == 0:
            return s.split(".")[0]

        if "." not in s:
            return s + "." + ("0" * digits)

        whole, frac = s.split(".", 1)

        # Pad or trim to exact length
        if len(frac) < digits:
            frac = frac + ("0" * (digits - len(frac)))
        elif len(frac) > digits:
            frac = frac[:digits]

        return whole + "." + frac

    def _quantize(self, value: Decimal, digits: int) -> Decimal:
        """
        Quantize to exactly `digits` digits after the decimal point.
        Uses ROUND_FLOOR (truncate for positive values).
        """
        q = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)  # 10^(-digits)
        return value.quantize(q, rounding=ROUND_FLOOR)

    def _quantize_safe(self, value: Decimal, digits: int, min_prec: int) -> Decimal:
        """
        Quantize with a guaranteed-large enough decimal context precision.

        This prevents decimal.InvalidOperation when quantize() is called outside
        the original high-precision localcontext().
        """
        needed = digits + self.guard_digits
        with localcontext() as ctx:
            ctx.prec = max(min_prec, needed, 50)  # 50 is a tiny floor to avoid silly small contexts
            return self._quantize(value, digits)


def _self_test():
    """
    Minimal sanity check:
    - First 50 digits should start with the known prefix.
    """
    eng = PiEngine()
    p50 = eng.pi_str(50)
    # Known prefix of π:
    known = "3.14159265358979323846264338327950288419716939937510"
    assert p50 == known, f"Mismatch:\n{p50}\n!=\n{known}"


if __name__ == "__main__":
    eng = PiEngine()

    print("π with 50 digits after decimal:")
    print(eng.pi_str(50))
    print()

    print("π with 200 digits after decimal:")
    print(eng.pi_str(200))
    print()

    # Demonstrate dynamic scaling + caching:
    # If you already computed 200 digits, asking for 80 should reuse the cached π.
    print("π with 80 digits after decimal (should reuse cached π):")
    print(eng.pi_str(80))
    print()

    # Optional quick self-test (comment out if you don't want asserts)
    _self_test()
    print("Self-test OK.")
