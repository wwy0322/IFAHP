from .base_model import BaseModel


class FakeModel(BaseModel):

    def __init__(self):
        super(FakeModel, self).__init__()

    def calc_membership(self, mem_i: float, mem_j: float) -> float:
        if super(FakeModel, self).calc_membership(mem_i, mem_j) < 0:
            return -1
        else:
            return (1 + mem_i - mem_j) / 2

    def calc_nonmembership(self, nmem_i: float, nmem_j: float) -> float:
        if super(FakeModel, self).calc_nonmembership(nmem_j, nmem_j) < 0:
            return -1
        else:
            return (1 + nmem_j - nmem_j) / 2
