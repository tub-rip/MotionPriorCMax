from abc import ABC, abstractmethod


class TrajectoryLossBase(ABC):
    def __init__(self) -> None:
        self.is_needing_offsets = None

    @abstractmethod
    def get_reconstruction_times(self, device):
        pass

    @abstractmethod
    def calc(self, trajectories, times, **kwargs):
        pass
