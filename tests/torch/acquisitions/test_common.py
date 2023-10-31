import pytest

import torch
from astra.torch.al import (
    UniformRandomAcquisition,
    RandomStrategy,
    EnsembleAcquisition,
    EnsembleStrategy,
    MCStrategy,
    MCAcquisition,
    DiversityStrategy,
    DiversityAcquisition,
    DeterministicAcquisition,
    DeterministicStrategy,
)
from astra.torch.al.errors import AcquisitionMismatchError


def test_acquisition_match():
    inputs = torch.rand(10, 3, 32, 32)
    outputs = torch.randint(0, 10, (10,))
    acquision = UniformRandomAcquisition()
    strategy = RandomStrategy(acquision, inputs, outputs)
    # If code reaches here, then the test passes


@pytest.mark.parametrize(
    "another_class", [EnsembleAcquisition, MCAcquisition, DiversityAcquisition, DeterministicAcquisition]
)
def test_acquisition_mismatch(another_class):
    inputs = torch.rand(10, 3, 32, 32)
    outputs = torch.randint(0, 10, (10,))

    class DummyAcquisition(another_class):
        def acquire_scores(self, logits):
            return logits

    another_acquisition = DummyAcquisition()

    # capture acquisition mismatch error
    with pytest.raises(AcquisitionMismatchError):
        strategy = RandomStrategy(another_acquisition, inputs, outputs)
