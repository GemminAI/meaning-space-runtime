"""MSR's own implementation of the IC-00 wire rules (design v0.6.2 §5-0, §7, §2.1).

Nothing here imports sensos or any neighbour; the rules are re-implemented from
the contract text and pinned by the language-neutral vectors vendored under
``tests/contract/vendor``.

* :func:`strict_load` — the Integration Gate's parsing stage: integer tokens
  only, no ``NaN``/``Infinity``, no duplicate keys (compared after unescaping),
  no lone surrogates, UTF-8 only.
* :func:`quantize` / :func:`dequantize` — ``q = round_ties_to_even(value * 2**-e)``.
  ``math.ldexp`` scales exactly and the built-in ``round()`` on a float is
  ties-to-even (design §7's ``numpy.rint`` remark is wrong; no NumPy needed).
* :func:`jcs` — RFC 8785 for integer-only values.
"""

from __future__ import annotations

import json
import math
from typing import Any, NoReturn

from msr.errors import MSRError

MAX_SAFE = 2**53 - 1
ENVELOPE_RANGE = (-MAX_SAFE, MAX_SAFE)
INT64_RANGE = (-(2**63), 2**63 - 1)

E_UTF8 = "E_UTF8"
E_SYNTAX = "E_SYNTAX"
E_SURROGATE = "E_SURROGATE"
E_FLOAT_TOKEN = "E_FLOAT_TOKEN"
E_CONSTANT = "E_CONSTANT"
E_DUP_KEY = "E_DUP_KEY"
E_INT_RANGE = "E_INT_RANGE"
E_FLOAT_TYPE = "E_FLOAT_TYPE"
E_NOT_CANONICALIZABLE = "E_NOT_CANONICALIZABLE"
E_QUANT_NONFINITE = "E_QUANT_NONFINITE"
E_QUANT_RANGE = "E_QUANT_RANGE"
E_PARENTS = "E_PARENTS"
E_LOOKUP = "E_LOOKUP"
E_INVARIANT = "E_INVARIANT"


class ContractError(MSRError):
    """A wire-level or envelope-level contract rule was violated."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _float_token(token: str) -> NoReturn:
    raise ContractError(
        E_FLOAT_TOKEN, f"number token {token!r} has a fraction/exponent"
    )


def _constant(token: str) -> NoReturn:
    raise ContractError(E_CONSTANT, f"{token} is not allowed")


def _lone_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if _lone_surrogate(key):
            raise ContractError(E_SURROGATE, "lone surrogate in an object key")
        if key in out:
            raise ContractError(E_DUP_KEY, f"duplicate key {key!r}")
        out[key] = value
    return out


def _check_strings(value: Any) -> None:
    if isinstance(value, str):
        if _lone_surrogate(value):
            raise ContractError(E_SURROGATE, "lone surrogate in a string")
    elif isinstance(value, list):
        for item in value:
            _check_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            _check_strings(item)


def strict_load(data: bytes) -> Any:
    """Parse JSON bytes under the Gate's step-0 rules (§5)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(E_UTF8, str(exc)) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=_float_token,
            parse_constant=_constant,
        )
    except ContractError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ContractError(E_SYNTAX, str(exc)) from exc
    _check_strings(value)
    return value


def quantize(
    value: float, exponent: int, *, int_range: tuple[int, int] = ENVELOPE_RANGE
) -> int:
    """``round_ties_to_even(value * 2**-exponent)``; refuse NaN/inf/out of range."""
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise ContractError(E_QUANT_NONFINITE, "value must be a float or int")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ContractError(E_QUANT_NONFINITE, "NaN and infinity cannot be quantized")
    try:
        scaled = math.ldexp(as_float, -exponent)
    except OverflowError as exc:
        raise ContractError(E_QUANT_RANGE, "value too large for the quantum") from exc
    q = round(scaled)  # float -> int: ties-to-even
    lo, hi = int_range
    if not lo <= q <= hi:
        raise ContractError(E_QUANT_RANGE, f"{q} outside [{lo}, {hi}]")
    return int(q)


def dequantize(
    q: int, exponent: int, *, int_range: tuple[int, int] = ENVELOPE_RANGE
) -> float:
    if isinstance(q, bool) or not isinstance(q, int):
        raise ContractError(E_QUANT_RANGE, "q must be an int")
    lo, hi = int_range
    if not lo <= q <= hi:
        raise ContractError(E_QUANT_RANGE, f"{q} outside [{lo}, {hi}]")
    return math.ldexp(float(q), exponent)


_SHORT = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _string(text: str) -> str:
    out = ['"']
    for ch in text:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            raise ContractError(E_SURROGATE, "lone surrogate in a string")
        if ch in _SHORT:
            out.append(_SHORT[ch])
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _utf16_key(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")


def canonicalize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        raise ContractError(E_FLOAT_TYPE, f"float {value!r} is not allowed")
    if isinstance(value, int):
        if abs(value) > MAX_SAFE:
            raise ContractError(E_INT_RANGE, f"{value} outside +-(2^53-1)")
        return str(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list | tuple):
        return "[" + ",".join(canonicalize(item) for item in value) + "]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise ContractError(E_NOT_CANONICALIZABLE, f"key {key!r} is not a str")
        keys = sorted(value, key=_utf16_key)
        return (
            "{" + ",".join(f"{_string(k)}:{canonicalize(value[k])}" for k in keys) + "}"
        )
    raise ContractError(E_NOT_CANONICALIZABLE, type(value).__name__)


def jcs(value: Any) -> bytes:
    """RFC 8785 canonical UTF-8 bytes of an integer-only JSON value."""
    return canonicalize(value).encode("utf-8")


__all__ = [
    "ENVELOPE_RANGE",
    "INT64_RANGE",
    "MAX_SAFE",
    "ContractError",
    "canonicalize",
    "dequantize",
    "jcs",
    "quantize",
    "strict_load",
]
