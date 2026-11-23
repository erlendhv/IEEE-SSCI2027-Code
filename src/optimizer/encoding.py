"""Gray code and standard binary encoding of particles as PortfolioAllocations.

In Gray code, adjacent integers differ by one bit, so a single bit flip
can move a decoded value to its neighbour.

Encoding per field (concatenated):
  [allocation bits | start_year bits | wells bits]

  - allocation: ceil(log2(n_hosts + 1)) bits, values 0..n_hosts
    where 0..n_hosts-1 = host index, n_hosts = not developed
  - start_year: ceil(log2(max_start_year + 1)) bits, values 0..max_start_year
  - wells: ceil(log2(max_wells)) bits, 1-indexed (decoded 0 = 1 well)

Total particle length = n_fields * (alloc_bits + year_bits + wells_bits)
"""

import math

import numpy as np

from src.models import PortfolioAllocation


# ── Gray code primitives ─────────────────────────────────────────


def int_to_gray(n: int) -> int:
    """Convert integer to Gray code value."""
    return n ^ (n >> 1)


def gray_to_int(gray: int) -> int:
    """Convert Gray code value back to integer."""
    n = gray
    mask = n >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def int_to_gray_binary(value: int, n_bits: int) -> np.ndarray:
    """Convert a non-negative integer to a Gray-coded bit array of length n_bits (MSB first)."""
    gray = int_to_gray(value)
    bits = np.array(
        [(gray >> (n_bits - 1 - i)) & 1 for i in range(n_bits)],
        dtype=int,
    )
    return bits


def gray_binary_to_int(bits: np.ndarray) -> int:
    """Convert a Gray-coded bit array (MSB first) back to an integer."""
    n_bits = len(bits)
    gray = sum(int(b) << (n_bits - 1 - i) for i, b in enumerate(bits))
    return gray_to_int(gray)


# ── Standard binary primitives ──────────────────────────────────


def int_to_binary(value: int, n_bits: int) -> np.ndarray:
    """Convert a non-negative integer to a standard binary bit array of length n_bits (MSB first)."""
    bits = np.array(
        [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)],
        dtype=int,
    )
    return bits


def binary_to_int(bits: np.ndarray) -> int:
    """Convert a standard binary bit array (MSB first) back to an integer."""
    n_bits = len(bits)
    return sum(int(b) << (n_bits - 1 - i) for i, b in enumerate(bits))


# ── Encoding dispatch helpers ───────────────────────────────────


def _decode_bits(bits: np.ndarray, encoding_type: str) -> int:
    """Decode a bit array with "gray" or "binary" encoding."""
    if encoding_type == "gray":
        return gray_binary_to_int(bits)
    return binary_to_int(bits)


def _encode_int(value: int, n_bits: int, encoding_type: str) -> np.ndarray:
    """Encode an integer with "gray" or "binary" encoding."""
    if encoding_type == "gray":
        return int_to_gray_binary(value, n_bits)
    return int_to_binary(value, n_bits)


# ── Bit-width helpers ────────────────────────────────────────────


def allocation_bits(n_hosts: int) -> int:
    """Number of bits needed to encode host assignment (0..n_hosts)."""
    return math.ceil(math.log2(n_hosts + 1))


def year_bits(max_start_year: int) -> int:
    """Number of bits needed to encode start year (0..max_start_year)."""
    return math.ceil(math.log2(max_start_year + 1))


def wells_bits(max_wells: int) -> int:
    """Number of bits needed to encode well count (1..max_wells)."""
    return math.ceil(math.log2(max_wells))


def particle_length(
    n_fields: int, n_hosts: int, max_start_year: int, max_wells: int,
) -> int:
    """Total binary particle length."""
    return n_fields * (
        allocation_bits(n_hosts) + year_bits(max_start_year) + wells_bits(max_wells)
    )


# ── Particle decoder ─────────────────────────────────────────────


def decode_particle(
    binary: np.ndarray,
    n_fields: int,
    n_hosts: int,
    max_start_year: int,
    max_wells: int,
    encoding: dict | None = None,
) -> PortfolioAllocation:
    """Decode a flat binary particle into a PortfolioAllocation.

    Decoded values are mapped to valid ranges:
      - allocation >= n_hosts: not developed (-1)
      - start_year > max_start_year: clamped to max_start_year
      - wells: 1-indexed (value 0 = 1 well), clamped to max_wells

    Args:
        binary: Flat array of 0s and 1s.
        max_start_year: Latest year a field can begin development.
        max_wells: Maximum wells per field.
        encoding: Dict with keys 'assignment', 'timing', 'wells', each
            "gray" or "binary". None means Gray code for all.

    Raises:
        ValueError: If the binary length does not match the dimensions.
    """
    enc_assign = "gray" if encoding is None else encoding["assignment"]
    enc_timing = "gray" if encoding is None else encoding["timing"]
    enc_wells = "gray" if encoding is None else encoding["wells"]

    n_alloc = allocation_bits(n_hosts)
    n_year = year_bits(max_start_year)
    n_wells_bits = wells_bits(max_wells)
    bits_per_field = n_alloc + n_year + n_wells_bits
    expected_len = n_fields * bits_per_field

    if len(binary) != expected_len:
        raise ValueError(
            f"Expected {expected_len} bits "
            f"({n_fields} fields × {bits_per_field} bits/field), "
            f"got {len(binary)}"
        )

    assignment = np.empty(n_fields, dtype=int)
    start_year = np.empty(n_fields, dtype=int)
    n_wells = np.empty(n_fields, dtype=int)

    for f in range(n_fields):
        offset = f * bits_per_field

        alloc_val = _decode_bits(
            binary[offset : offset + n_alloc], enc_assign,
        )
        if alloc_val >= n_hosts:
            assignment[f] = -1
        else:
            assignment[f] = alloc_val

        year_val = _decode_bits(
            binary[offset + n_alloc : offset + n_alloc + n_year],
            enc_timing,
        )
        start_year[f] = min(year_val, max_start_year)

        wells_val = _decode_bits(
            binary[offset + n_alloc + n_year : offset + bits_per_field],
            enc_wells,
        )
        n_wells[f] = min(wells_val + 1, max_wells)

    return PortfolioAllocation(
        assignment=assignment, start_year=start_year, n_wells=n_wells,
    )


# ── Particle encoder ─────────────────────────────────────────────


def encode_allocation(
    alloc: PortfolioAllocation,
    n_hosts: int,
    max_start_year: int,
    max_wells: int,
    encoding: dict | None = None,
) -> np.ndarray:
    """Encode a PortfolioAllocation into a flat binary particle (inverse of decode_particle).

    Args:
        encoding: Dict with keys 'assignment', 'timing', 'wells', each
            "gray" or "binary". None means Gray code for all.
    """
    enc_assign = "gray" if encoding is None else encoding["assignment"]
    enc_timing = "gray" if encoding is None else encoding["timing"]
    enc_wells = "gray" if encoding is None else encoding["wells"]

    n_alloc = allocation_bits(n_hosts)
    n_year = year_bits(max_start_year)
    n_wells_bits = wells_bits(max_wells)
    n_fields = alloc.n_fields

    parts = []
    for f in range(n_fields):
        host = alloc.assignment[f]
        alloc_val = n_hosts if host == -1 else int(host)
        parts.append(_encode_int(alloc_val, n_alloc, enc_assign))

        year_val = int(alloc.start_year[f])
        parts.append(_encode_int(year_val, n_year, enc_timing))

        wells_val = int(alloc.n_wells[f]) - 1
        parts.append(_encode_int(wells_val, n_wells_bits, enc_wells))

    return np.concatenate(parts)
