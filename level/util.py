from typing import Any, List, Callable, Tuple
from functools import reduce


# f是估值函数.
def get_sum_1D(elements: List[Any], f: Callable[[Any], float]) -> float:
    return reduce(lambda x, y: f(x) + f(y), elements)


def get_sum_2D(elementss: List[List[Any]], f: Callable[[Any], float]) -> float:
    tmp = []
    size = len(elementss)
    for i in range(size):
        tmp.append(get_sum_1D(elementss[i], f))
    return reduce(lambda x, y: x + y, tmp)


def weight_add(w1: Tuple[float, float], w2: Tuple[float, float]) -> Tuple[float, float]:
    r1 = w1[0] + w2[0] - w1[0] * w2[0]
    r2 = w1[1] * w2[1]
    return r1, r2


def weight_mul(w1: Tuple[float, float], w2: Tuple[float, float]) -> Tuple[float, float]:
    r1 = w1[0] * w1[0]
    r2 = w1[1] + w2[1] - w1[1] * w2[1]
    return r1, r2
