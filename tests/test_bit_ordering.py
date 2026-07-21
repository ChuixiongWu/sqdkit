import numpy as np
import pytest
from qiskit.primitives import BitArray
from qiskit_aer import AerSimulator

from sqdkit.stateprep import ReferenceProductStatePrep, to_sqd_bit_array


def test_to_sqd_bit_array_preserves_spin_block_populations() -> None:
    # Given: alpha[0:norb),beta[norb:2norb) HF occupations measured once.
    stateprep = ReferenceProductStatePrep(norb=3, n_alpha=2, n_beta=1)
    circuit = stateprep.build(problem=None).circuit
    measured_circuit = circuit.copy()
    measured_circuit.measure_all()
    result = AerSimulator(method="statevector").run(
        measured_circuit, shots=1, memory=True, seed_simulator=7
    ).result()
    bit_array = BitArray.from_samples(list(result.get_counts()), num_bits=6)

    # When: spin_blocks:alpha[0:norb),beta[norb:2norb) become SQD [beta...,alpha...].
    sqd_samples = to_sqd_bit_array(bit_array, norb=3)

    # Then: each SQD spin half retains its documented electron population.
    assert np.count_nonzero(sqd_samples[0, :3]) == 1
    assert np.count_nonzero(sqd_samples[0, 3:]) == 2


def test_to_sqd_bit_array_rejects_wrong_width() -> None:
    # Given: a measured sample whose width does not match two spatial orbitals.
    bit_array = BitArray.from_samples(["101"])

    # When / Then: conversion rejects the incompatible spin-block width.
    with pytest.raises(ValueError, match="width"):
        to_sqd_bit_array(bit_array, norb=2)
