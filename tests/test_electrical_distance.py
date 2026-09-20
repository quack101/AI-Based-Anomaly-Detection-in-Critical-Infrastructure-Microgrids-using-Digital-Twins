"""
Verification of placement/electrical_distance.py against XuIoT26's
worked examples (Section II-B, eq. 33-38). These numbers are hardcoded
from the paper, not derived -- this is a formula-correctness check, not
a regression test.

Our feeder is the unmodified IEEE-123; XuIoT26's is the same feeder
with DGs added, so exact bit-for-bit agreement is not expected -- only
agreement to the paper's stated rounding.
"""

import pytest

from core.ieee123_model import extract_all
from placement.electrical_distance import (
    get_system_y_index,
    bus_pair_distance_matrix,
    bus_pair_distance,
)

# XuIoT26 eq. (33)-(38): three-phase line, buses 1-7.
PAPER_D17_MATRIX = [
    [0.97, 0.36, 0.21],
    [0.29, 0.80, 0.22],
    [0.18, 0.23, 0.83],
]
PAPER_D17_SUM = 4.09
PAPER_D17_DIAGONAL_SHARE_PCT = 63.6

# XuIoT26: single-phase (phase b) line, buses 1-2.
PAPER_D12_BB = 1.31


@pytest.fixture(scope="module")
def y_index():
    extract_all()  # compiles + solves; leaves the circuit active
    return get_system_y_index()


def test_three_phase_worked_example_d17(y_index):
    Y, node_index = y_index
    matrix = bus_pair_distance_matrix(Y, node_index, "1", "7")

    for row_idx in range(3):
        for col_idx in range(3):
            got = matrix[row_idx][col_idx]
            expected = PAPER_D17_MATRIX[row_idx][col_idx]
            assert got == pytest.approx(expected, abs=0.02), (
                f"D_17[{row_idx}][{col_idx}] = {got:.4f}, paper = {expected}"
            )

    total = sum(sum(row) for row in matrix)
    assert total == pytest.approx(PAPER_D17_SUM, abs=0.02)

    diagonal_share = 100.0 * sum(matrix[i][i] for i in range(3)) / total
    assert diagonal_share == pytest.approx(PAPER_D17_DIAGONAL_SHARE_PCT, abs=0.5)


def test_single_phase_worked_example_d12(y_index):
    Y, node_index = y_index
    matrix = bus_pair_distance_matrix(Y, node_index, "1", "2")

    # phase b is index 1 (phases enumerated 1,2,3 -> a,b,c)
    assert matrix[1][1] == pytest.approx(PAPER_D12_BB, abs=0.02)

    for row_idx in range(3):
        for col_idx in range(3):
            if (row_idx, col_idx) == (1, 1):
                continue
            assert matrix[row_idx][col_idx] == pytest.approx(0.0, abs=1e-9)

    assert bus_pair_distance(Y, node_index, "1", "2") == pytest.approx(
        PAPER_D12_BB, abs=0.02
    )


def test_distance_is_symmetric(y_index):
    Y, node_index = y_index
    assert bus_pair_distance(Y, node_index, "1", "7") == pytest.approx(
        bus_pair_distance(Y, node_index, "7", "1")
    )


def test_non_adjacent_buses_have_zero_distance(y_index):
    Y, node_index = y_index
    # bus 1 and bus 300 are far apart, no direct branch
    assert bus_pair_distance(Y, node_index, "1", "300") == 0.0
