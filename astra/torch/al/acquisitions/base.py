from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Sequence, Dict


# Base classes for acquisition functions


class Acquisition(ABC):
    @abstractmethod
    def acquire_scores(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass")


class DeterministicAcquisition(Acquisition):
    @abstractmethod
    def acquire_scores(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass")


class RandomAcquisition(Acquisition):
    @abstractmethod
    def acquire_scores(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass")


class MCAcquisition(Acquisition):
    @abstractmethod
    def acquire_scores(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass")


class EnsembleAcquisition(Acquisition):
    @abstractmethod
    def acquire_scores(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass")


class DiversityAcquisition(Acquisition):
    @abstractmethod
    def acquire_scores(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass")
