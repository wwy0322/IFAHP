import abc


class BaseModel:
    @abc.abstractmethod
    def __init__(self):
        return

    @abc.abstractmethod
    def calc_membership(self, mem_i: float, mem_j: float) -> float:
        if mem_i < 0 or mem_i > 1:
            return -1
        if mem_j < 0 or mem_j > 1:
            return -1
        return 1

    @abc.abstractmethod
    def calc_nonmembership(self, nmem_i: float, nmem_j: float) -> float:
        if nmem_i < 0 or nmem_i > 1:
            return -1
        if nmem_j < 0 or nmem_j > 1:
            return -1
        return 1
