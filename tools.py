import numpy as np


def sample_y_coords_with_weighted_intervals(rng, y_min, y_max, num_random_points, K=5):
    """
    在 [y_min, y_max) 中生成整数，其中前 1/5 区间内每个点的概率密度是其余区间的 K 倍。
    
    参数:
        rng: numpy.random.Generator 实例
        y_min, y_max: 整数，定义采样范围（左闭右开）
        num_random_points: 要生成的点数
        K: 高概率区间的密度倍数（默认 3）
    """
    total_range = y_max - y_min

    split_point = y_min + total_range // 5

    len_A = split_point - y_min
    len_B = y_max - split_point

    if len_A <= 0:
        return rng.integers(y_min, y_max, size=num_random_points)
    if len_B <= 0:
        return rng.integers(y_min, split_point, size=num_random_points)

    p_A = (K * len_A) / (K * len_A + len_B)

    from_A = rng.random(num_random_points) < p_A

    y_coords = np.empty(num_random_points, dtype=int)

    n_A = from_A.sum()
    if n_A > 0:
        y_coords[from_A] = rng.integers(y_min, split_point, size=n_A)

    n_B = (~from_A).sum()
    if n_B > 0:
        y_coords[~from_A] = rng.integers(split_point, y_max, size=n_B)

    return y_coords