"""Tests for Gray code encoding and particle decoding."""

import math

import numpy as np
import pytest

from src.optimizer.encoding import (
    allocation_bits,
    binary_to_int,
    decode_particle,
    encode_allocation,
    gray_binary_to_int,
    int_to_binary,
    int_to_gray_binary,
    particle_length,
    wells_bits,
    year_bits,
)
from src.models import PortfolioAllocation


# ── Gray code primitives ─────────────────────────────────────────

class TestGrayCode:
    """Test Gray code encode/decode roundtrip and properties."""

    def test_roundtrip_4bit(self):
        """Encode then decode should return original value."""
        n_bits = 4
        for value in range(16):
            gray = int_to_gray_binary(value, n_bits)
            decoded = gray_binary_to_int(gray)
            assert decoded == value, (
                f"Failed for {value}: encoded={gray}, decoded={decoded}"
            )

    def test_roundtrip_various_bit_widths(self):
        """Roundtrip works for different bit widths."""
        for n_bits in [1, 2, 3, 5, 8]:
            max_val = 2**n_bits
            for value in range(max_val):
                gray = int_to_gray_binary(value, n_bits)
                assert gray_binary_to_int(gray) == value

    def test_adjacent_differ_by_one_bit(self):
        """Adjacent integers should differ by exactly one bit in Gray code."""
        n_bits = 4
        for value in range(15):
            g1 = int_to_gray_binary(value, n_bits)
            g2 = int_to_gray_binary(value + 1, n_bits)
            diff = np.sum(g1 != g2)
            assert diff == 1, (
                f"Values {value} and {value+1} differ by {diff} bits"
            )

    def test_zero_is_all_zeros(self):
        """Gray code of 0 should be all zeros."""
        for n_bits in [1, 2, 4, 8]:
            gray = int_to_gray_binary(0, n_bits)
            assert np.all(gray == 0)

    def test_output_shape_and_dtype(self):
        """Output should be an int array of the correct length."""
        gray = int_to_gray_binary(5, 4)
        assert gray.shape == (4,)
        assert gray.dtype == int


# ── Particle decoding ────────────────────────────────────────────

# Standard test dimensions: 2 alloc bits + 4 year bits + 3 wells bits = 9 per field
N_HOSTS = 2
MAX_START_YEAR = 14
MAX_WELLS = 8
N_ALLOC = allocation_bits(N_HOSTS)
N_YEAR = year_bits(MAX_START_YEAR)
N_WELLS = wells_bits(MAX_WELLS)
BITS_PER_FIELD = N_ALLOC + N_YEAR + N_WELLS


def _make_field_bits(host_val, year_val, wells_val):
    """Build Gray-coded bits for one field: [alloc | year | wells]."""
    return np.concatenate([
        int_to_gray_binary(host_val, N_ALLOC),
        int_to_gray_binary(year_val, N_YEAR),
        int_to_gray_binary(wells_val, N_WELLS),
    ])


class TestDecodeParticle:
    """Test binary particle → PortfolioAllocation."""

    def test_all_zeros(self):
        """All-zero particle: host 0, start year 0, 1 well."""
        n_fields = 3
        particle = np.zeros(n_fields * BITS_PER_FIELD, dtype=int)
        alloc = decode_particle(
            particle, n_fields, N_HOSTS, MAX_START_YEAR, MAX_WELLS,
        )

        assert alloc.assignment.shape == (3,)
        assert alloc.start_year.shape == (3,)
        assert alloc.n_wells.shape == (3,)
        np.testing.assert_array_equal(alloc.assignment, [0, 0, 0])
        np.testing.assert_array_equal(alloc.start_year, [0, 0, 0])
        # Gray(0) = 0 → wells_val=0 → n_wells=1
        np.testing.assert_array_equal(alloc.n_wells, [1, 1, 1])

    def test_not_developed_mapping(self):
        """Allocation value == n_hosts should map to -1 (not developed)."""
        particle = _make_field_bits(2, 0, 0)
        alloc = decode_particle(particle, 1, N_HOSTS, MAX_START_YEAR, MAX_WELLS)
        assert alloc.assignment[0] == -1

    def test_invalid_allocation_maps_to_not_developed(self):
        """Decoded allocation > n_hosts should map to not developed."""
        # 2 bits can represent 0-3, value 3 > n_hosts → not developed
        particle = _make_field_bits(3, 0, 0)
        alloc = decode_particle(particle, 1, N_HOSTS, MAX_START_YEAR, MAX_WELLS)
        assert alloc.assignment[0] == -1

    def test_start_year_clamping(self):
        """Start year > max_start_year should be clamped to max_start_year."""
        # 4 bits can represent 0-15, value 15 > 14 → clamped to 14
        particle = _make_field_bits(0, 15, 0)
        alloc = decode_particle(particle, 1, N_HOSTS, MAX_START_YEAR, MAX_WELLS)
        assert alloc.start_year[0] == 14

    def test_wells_decoding(self):
        """Wells are 1-indexed: decoded value 0 → 1 well, 4 → 5 wells."""
        particle = _make_field_bits(0, 0, 4)
        alloc = decode_particle(particle, 1, N_HOSTS, MAX_START_YEAR, MAX_WELLS)
        assert alloc.n_wells[0] == 5

    def test_wells_clamped_to_max(self):
        """The largest 3-bit wells value (7) decodes to max_wells (8)."""
        particle = _make_field_bits(0, 0, 7)
        alloc = decode_particle(particle, 1, N_HOSTS, MAX_START_YEAR, MAX_WELLS)
        assert alloc.n_wells[0] == MAX_WELLS

    def test_known_encoding(self):
        """Decode a hand-built two-field particle."""
        # Field 0: host 1, start year 5, 3 wells (wells_val=2)
        # Field 1: not developed (host_val = n_hosts), start year 3, 7 wells (wells_val=6)
        f0 = _make_field_bits(1, 5, 2)
        f1 = _make_field_bits(2, 3, 6)
        particle = np.concatenate([f0, f1])
        alloc = decode_particle(particle, 2, N_HOSTS, MAX_START_YEAR, MAX_WELLS)

        assert alloc.assignment[0] == 1
        assert alloc.assignment[1] == -1
        assert alloc.start_year[0] == 5
        assert alloc.start_year[1] == 3
        assert alloc.n_wells[0] == 3
        assert alloc.n_wells[1] == 7

    def test_particle_length_validation(self):
        """Wrong-length particle should raise ValueError."""
        with pytest.raises(ValueError, match="Expected .* bits"):
            decode_particle(
                np.zeros(10, dtype=int), 3, N_HOSTS, MAX_START_YEAR, MAX_WELLS,
            )


# ── Encode allocation (reverse direction) ────────────────────────

class TestEncodeAllocation:
    """Test PortfolioAllocation → binary particle."""

    def test_roundtrip(self):
        """encode → decode should recover the original allocation."""
        alloc = PortfolioAllocation(
            assignment=np.array([0, 1, -1]),
            start_year=np.array([3, 7, 0]),
            n_wells=np.array([5, 3, 1]),
        )
        binary = encode_allocation(alloc, N_HOSTS, MAX_START_YEAR, MAX_WELLS)
        recovered = decode_particle(binary, 3, N_HOSTS, MAX_START_YEAR, MAX_WELLS)

        np.testing.assert_array_equal(recovered.assignment, alloc.assignment)
        np.testing.assert_array_equal(recovered.start_year, alloc.start_year)
        np.testing.assert_array_equal(recovered.n_wells, alloc.n_wells)

    def test_not_developed_encodes_as_n_hosts(self):
        """Assignment -1 should encode as value n_hosts."""
        alloc = PortfolioAllocation(
            assignment=np.array([-1]),
            start_year=np.array([0]),
            n_wells=np.array([1]),
        )
        binary = encode_allocation(alloc, N_HOSTS, MAX_START_YEAR, MAX_WELLS)
        assert gray_binary_to_int(binary[:N_ALLOC]) == N_HOSTS

    def test_wells_roundtrip_all_values(self):
        """All valid well counts (1..max_wells) survive encode → decode."""
        for w in range(1, MAX_WELLS + 1):
            alloc = PortfolioAllocation(
                assignment=np.array([0]),
                start_year=np.array([0]),
                n_wells=np.array([w]),
            )
            binary = encode_allocation(alloc, N_HOSTS, MAX_START_YEAR, MAX_WELLS)
            recovered = decode_particle(
                binary, 1, N_HOSTS, MAX_START_YEAR, MAX_WELLS,
            )
            assert recovered.n_wells[0] == w, f"Failed for n_wells={w}"


# ── Bit width helpers ────────────────────────────────────────────

class TestBitWidths:
    """Bit widths: ceil(log2(n_hosts + 1)), ceil(log2(max_start_year + 1)) and ceil(log2(max_wells))."""

    def test_small_problem(self):
        """3 fields, 2 hosts, max_start_year=14, max_wells=8."""
        n_fields = 3
        ab = allocation_bits(2)
        yb = year_bits(14)
        wb = wells_bits(8)
        total = n_fields * (ab + yb + wb)
        assert ab == 2
        assert yb == 4
        assert wb == 3
        assert total == 27

    def test_larger_problem(self):
        """5 fields, 4 hosts, max_start_year=19, max_wells=25."""
        n_fields = 5
        ab = allocation_bits(4)
        yb = year_bits(19)
        wb = wells_bits(25)
        total = n_fields * (ab + yb + wb)
        assert ab == 3
        assert yb == 5
        assert wb == 5
        assert total == 65

    def test_particle_length_function(self):
        """particle_length matches manual calculation."""
        pl = particle_length(20, 5, 6, 25)
        # alloc=3, year=3, wells=5 → 11 per field × 20 = 220
        assert pl == 20 * (3 + 3 + 5)


# ── Standard binary primitives ──────────────────────────────────

class TestStandardBinary:
    """Test standard binary encode/decode roundtrip and properties."""

    def test_roundtrip_4bit(self):
        """Encode then decode should return original value."""
        for value in range(16):
            bits = int_to_binary(value, 4)
            decoded = binary_to_int(bits)
            assert decoded == value

    def test_roundtrip_various_bit_widths(self):
        """Roundtrip works for different bit widths."""
        for n_bits in [1, 2, 3, 5, 8]:
            for value in range(2**n_bits):
                bits = int_to_binary(value, n_bits)
                assert binary_to_int(bits) == value

    def test_zero_is_all_zeros(self):
        """Binary of 0 should be all zeros."""
        for n_bits in [1, 2, 4, 8]:
            bits = int_to_binary(0, n_bits)
            assert np.all(bits == 0)

    def test_known_values(self):
        """Check specific known binary representations."""
        # 5 in 4 bits = 0101
        np.testing.assert_array_equal(int_to_binary(5, 4), [0, 1, 0, 1])
        # 7 in 3 bits = 111
        np.testing.assert_array_equal(int_to_binary(7, 3), [1, 1, 1])

    def test_differs_from_gray(self):
        """Standard binary and Gray code differ for most values."""
        # Value 2 in 3 bits: binary=010, Gray=011
        std = int_to_binary(2, 3)
        gray = int_to_gray_binary(2, 3)
        assert not np.array_equal(std, gray)

    def test_output_shape_and_dtype(self):
        """Output should be an int array of the correct length."""
        bits = int_to_binary(5, 4)
        assert bits.shape == (4,)
        assert bits.dtype == int


# ── Configurable encoding per variable type ─────────────────────

def _make_field_bits_with_encoding(host_val, year_val, wells_val, encoding):
    """Build bits for one field using specified encoding per variable type."""
    enc_fn = {
        "gray": int_to_gray_binary,
        "binary": int_to_binary,
    }
    return np.concatenate([
        enc_fn[encoding["assignment"]](host_val, N_ALLOC),
        enc_fn[encoding["timing"]](year_val, N_YEAR),
        enc_fn[encoding["wells"]](wells_val, N_WELLS),
    ])


ALL_GRAY = {"assignment": "gray", "timing": "gray", "wells": "gray"}
ALL_BINARY = {"assignment": "binary", "timing": "binary", "wells": "binary"}


class TestConfigurableEncoding:
    """Test per-variable-type encoding (Gray vs standard binary)."""

    def test_all_gray_matches_default(self):
        """Explicit all-Gray encoding produces same result as default (None)."""
        particle = _make_field_bits(1, 5, 2)
        alloc_default = decode_particle(
            particle, 1, N_HOSTS, MAX_START_YEAR, MAX_WELLS,
        )
        alloc_gray = decode_particle(
            particle, 1, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=ALL_GRAY,
        )
        np.testing.assert_array_equal(
            alloc_default.assignment, alloc_gray.assignment,
        )
        np.testing.assert_array_equal(
            alloc_default.start_year, alloc_gray.start_year,
        )
        np.testing.assert_array_equal(alloc_default.n_wells, alloc_gray.n_wells)

    def test_all_binary_roundtrip(self):
        """Encode → decode roundtrip with all-binary encoding."""
        alloc = PortfolioAllocation(
            assignment=np.array([0, 1, -1]),
            start_year=np.array([3, 7, 0]),
            n_wells=np.array([5, 3, 1]),
        )
        binary = encode_allocation(
            alloc, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=ALL_BINARY,
        )
        recovered = decode_particle(
            binary, 3, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=ALL_BINARY,
        )
        np.testing.assert_array_equal(recovered.assignment, alloc.assignment)
        np.testing.assert_array_equal(recovered.start_year, alloc.start_year)
        np.testing.assert_array_equal(recovered.n_wells, alloc.n_wells)

    def test_mixed_encoding_roundtrip(self):
        """Encode → decode roundtrip with mixed encoding."""
        mixed = {"assignment": "binary", "timing": "gray", "wells": "binary"}
        alloc = PortfolioAllocation(
            assignment=np.array([0, 1, -1]),
            start_year=np.array([3, 7, 0]),
            n_wells=np.array([5, 3, 1]),
        )
        binary = encode_allocation(
            alloc, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=mixed,
        )
        recovered = decode_particle(
            binary, 3, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=mixed,
        )
        np.testing.assert_array_equal(recovered.assignment, alloc.assignment)
        np.testing.assert_array_equal(recovered.start_year, alloc.start_year)
        np.testing.assert_array_equal(recovered.n_wells, alloc.n_wells)

    def test_different_encoding_produces_different_bits(self):
        """Gray and binary encoding produce different bit patterns for same allocation."""
        alloc = PortfolioAllocation(
            assignment=np.array([1]),
            start_year=np.array([5]),
            n_wells=np.array([3]),
        )
        gray_bits = encode_allocation(
            alloc, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=ALL_GRAY,
        )
        binary_bits = encode_allocation(
            alloc, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=ALL_BINARY,
        )
        assert not np.array_equal(gray_bits, binary_bits)

    def test_mismatched_encoding_gives_wrong_result(self):
        """Encoding with Gray then decoding with binary gives wrong values."""
        alloc = PortfolioAllocation(
            assignment=np.array([1]),
            start_year=np.array([5]),
            n_wells=np.array([3]),
        )
        bits = encode_allocation(
            alloc, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=ALL_GRAY,
        )
        wrong = decode_particle(
            bits, 1, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=ALL_BINARY,
        )
        # Gray code of start_year 5 is 0111, which binary decoding reads as 7
        differs = (
            wrong.assignment[0] != alloc.assignment[0]
            or wrong.start_year[0] != alloc.start_year[0]
            or wrong.n_wells[0] != alloc.n_wells[0]
        )
        assert differs

    def test_all_binary_wells_roundtrip_all_values(self):
        """All valid well counts survive encode → decode with binary encoding."""
        for w in range(1, MAX_WELLS + 1):
            alloc = PortfolioAllocation(
                assignment=np.array([0]),
                start_year=np.array([0]),
                n_wells=np.array([w]),
            )
            bits = encode_allocation(
                alloc, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=ALL_BINARY,
            )
            recovered = decode_particle(
                bits, 1, N_HOSTS, MAX_START_YEAR, MAX_WELLS, encoding=ALL_BINARY,
            )
            assert recovered.n_wells[0] == w, f"Failed for n_wells={w}"
