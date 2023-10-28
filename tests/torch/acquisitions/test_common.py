import pytest

import torch
from astra.torch.al import (
    RandomAcquisition,
    RandomStrategy,
    EnsembleAcquisition,
    EnsembleStrategy,
    MCStrategy,
    MCAcquisition,
    DiversityStrategy,
    DiversityAcquisition,
)
from astra.torch.al.errors import AcquisitionMismatchError


def test_mutual_pairs():
    acquision = RandomAcquisition()
    inputs = torch.rand(10, 3, 32, 32)
    outputs = torch.randint(0, 10, (10,))
    strategy = RandomStrategy(acquision, inputs, outputs)

    class DummyAcquisition(EnsembleAcquisition):
        def acquire_scores(self, logits):
            return logits

    another_acquisition = DummyAcquisition()

    # capture acquisition mismatch error
    with pytest.raises(AcquisitionMismatchError):
        strategy = RandomStrategy(another_acquisition, inputs, outputs)
