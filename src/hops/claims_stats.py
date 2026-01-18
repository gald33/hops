import math
import random
from typing import List, Optional, Tuple


try:
    from scipy.stats import spearmanr as _spearmanr  # type: ignore
except Exception:  # pragma: no cover
    _spearmanr = None


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def bootstrap_ci(
    values: List[float],
    rng_seed: int = 0,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    values = [float(v) for v in values if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(rng_seed)
    boot = []
    for _ in range(max(1, n_boot)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(_mean(sample))
    boot.sort()
    lo_idx = int(((1.0 - ci) / 2.0) * len(boot))
    hi_idx = int(((1.0 + ci) / 2.0) * len(boot))
    lo_idx = max(0, min(lo_idx, len(boot) - 1))
    hi_idx = max(0, min(hi_idx, len(boot) - 1))
    return boot[lo_idx], _mean(values), boot[hi_idx]


def paired_seed_bootstrap(
    pairs: List[Tuple[float, float]],
    rng_seed: int = 0,
    n_boot: int = 1000,
    ci: float = 0.95,
    effect: str = "delta",
) -> Tuple[float, float, float]:
    effect_values = []
    for treat, control in pairs:
        if not (math.isfinite(treat) and math.isfinite(control)):
            continue
        if effect == "ratio":
            if control == 0:
                continue
            effect_values.append(treat / control)
        else:
            effect_values.append(treat - control)
    return bootstrap_ci(effect_values, rng_seed=rng_seed, n_boot=n_boot, ci=ci)


def _rankdata(values: List[float]) -> List[float]:
    sorted_vals = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1][0] == sorted_vals[i][0]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_vals[k][1]] = rank
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    if not x or not y:
        return float("nan")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def spearman_trend(x: List[float], y: List[float]) -> Tuple[float, Optional[float]]:
    if len(x) != len(y) or len(x) < 2:
        return float("nan"), None
    if _spearmanr is not None:
        rho, pvalue = _spearmanr(x, y)
        return float(rho), float(pvalue)
    rx = _rankdata(x)
    ry = _rankdata(y)
    rho = _pearson(rx, ry)
    return float(rho), None
