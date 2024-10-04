from itertools import combinations, product
from typing import List

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .loss import covering_radius, covering_radius_upper_bound, packing_density_loss


def greedy_sorting_init(points: List[np.ndarray], init: int, start: List[np.ndarray]):
    """Use greedy algorithm to sort. This will be used as initialization for MILP algorithm.

    Args:
        points (List[np.ndarray]): Points to be sorted.
        init (int): The first index to choose from points.
        start (List[np.ndarray]): Points that are already sorted.

    Returns:
        Array: Sorted points
    """
    N_s = [len(l) for l in points]
    l = len(start)
    cons = np.concatenate(points)
    cons = np.concatenate([start, cons]) if l > 0 else cons
    dis = np.arccos(np.clip(np.abs(cons @ cons.T), -1, 1))
    cum = np.cumsum(N_s).tolist()
    cum.insert(0, 0)
    ind = [init]
    N_s[0] -= 1
    radius = np.pi / 2

    for i, n in enumerate(N_s):
        indices = set(j for j in range(cum[i], cum[i + 1]))
        if i == 0:
            indices.discard(ind[0])
        while len(indices) > 0:
            pos, mx = -1, -1
            for j in indices:
                cr = min(map(lambda x: dis[x + l, j + l], ind), default=0)
                if l > 0:
                    cr = min(cr, min(map(lambda x: dis[x, j + l], range(l)), default=0))
                cr = min(cr, radius)
                if cr > mx:
                    mx, pos = cr, j
            ind.append(pos)
            radius = min(radius, mx)
            indices.discard(pos)

    result = []
    for i in range(len(N_s)):
        li = []
        for j in range(cum[i], cum[i + 1]):
            li.append(cons[ind[j] + l])
        result.append(np.array(li))

    return result


def greedy_sorting(points: List[np.ndarray], start: List[np.ndarray]):
    """Greedy sort. This adds a step of iterating through the first point chosen.

    Args:
        points (List[np.ndarray]): Points to sort.
        start (List[np.ndarray]): Points that are already sorted.

    Returns:
        Array: Sorted points
    """
    res, mx = None, -1
    for i in range(len(points[0])):
        tmp = greedy_sorting_init(points, i, start)
        loss = packing_density_loss(np.concatenate(tmp), start)
        if loss > mx:
            res, mx = tmp, loss
    return res


def incremental_sorting_single_shell_init(
    points: np.ndarray,
    K: int,
    start: bool = False,
    time_limit: float = 600,
    output_flag: int = 1,
):
    """Incremental algorithm for sorting points. This function sorts the first K points from entire point set.

    Args:
        points (List[np.ndarray]): Points to sort.
        K (int): Number of points to sort
        start (bool, optional): Whether to use existing order as initialization. Defaults to False.
        time_limit (int, optional): Time limit for GUROBI to run. Defaults to 600.
        output_flag (int, optional): GUROBI output flag. Defaults to 1.

    Returns:
        Array: Array shaped (K, 3), sorted K points
    """
    N = len(points)
    M = 2
    eps = 1e-5

    dis = np.clip(np.abs(points @ points.T), -1, 1)

    m = gp.Model("optimalSortingSingleShell")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    x = m.addVars(N, K, lb=0, ub=1, vtype=GRB.BINARY, name="x")
    cos_theta = m.addVars(
        [i for i in np.arange(2, K + 1)],
        lb=np.max(
            [
                np.zeros((K - 1,)),
                np.cos(covering_radius_upper_bound(2 * np.arange(2, K + 1))),
            ],
            axis=0,
        ),
        ub=1,
        vtype=GRB.CONTINUOUS,
        name="theta",
    )

    m.update()
    if start:
        for i, j in product(range(N), range(K)):
            m.getVarByName(f"x[{i},{j}]").Start = 1 if i == j else 0

    m.addConstrs((x.sum("*", j) == 1 for j in range(K)))
    for i in range(N):
        m.addSOS(GRB.SOS_TYPE1, [x[i, j] for j in range(K)])
    for j in range(K):
        m.addSOS(GRB.SOS_TYPE1, [x[i, j] for i in range(N)])

    for k in range(2, K + 1):
        m.addConstrs(
            (
                cos_theta[k]
                >= dis[i][j]
                - (
                    2
                    - gp.LinExpr([1 for _ in range(k)], [x[i, l] for l in range(k)])
                    - gp.LinExpr([1 for _ in range(k)], [x[j, l] for l in range(k)])
                )
                * M
                for i, j in combinations(range(N), 2)
            ),
            name="c",
        )
    m.addConstrs(cos_theta[k] <= cos_theta[k + 1] for k in range(2, K))

    m.setObjective(
        gp.LinExpr(
            [i for i in range(2, K + 1)], [cos_theta[i] for i in range(2, K + 1)]
        ),
        GRB.MINIMIZE,
    )

    m.optimize()

    l = [None for _ in range(K)]
    for i, j in product(range(N), range(K)):
        if np.abs(m.getVarByName(f"x[{i},{j}]").X - 1) < eps:
            l[j] = points[i]

    return l


def incremental_sorting_single_shell_incre(
    fixed_points: np.ndarray,
    incre_points: np.ndarray,
    num: int,
    start=False,
    time_limit: float = 600,
    output_flag: int = 1,
):
    """Incremental algorithm for sorting points. This function sorts the next num points from incre_points with fixed_points already sorted

    Args:
        fixed_points (np.ndarray): Points that are already sorted
        incre_points (np.ndarray): Points to sort.
        num (int): Number of points to sort
        start (bool, optional): Whether to use existing order as initialization. Defaults to False.
        time_limit (int, optional): Time limit for GUROBI to run. Defaults to 600.
        output_flag (int, optional): GUROBI output flag. Defaults to 1.

    Returns:
        Array: Array shaped (num, 3), sorted num points
    """
    N = len(incre_points)
    K = len(fixed_points)
    M = np.pi / 2
    eps = 1e-5

    points = np.concatenate([incre_points, fixed_points])
    dis = np.clip(np.abs(points @ points.T), -1, 1)
    fixed_angle = np.cos(covering_radius(fixed_points))
    lb = [
        max(fixed_angle, np.cos(covering_radius_upper_bound(n)))
        for n in range(K + 1, K + num + 1)
    ]

    m = gp.Model("optimalSortingSingleShell")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    x = m.addVars(N, num, lb=0, ub=1, vtype=GRB.BINARY, name="x")
    cos_theta = m.addVars(
        [i for i in range(1, num + 1)],
        lb=lb,
        ub=1,
        vtype=GRB.CONTINUOUS,
        name="theta",
    )

    m.update()
    if start:
        for i, j in product(range(N), range(num)):
            m.getVarByName(f"x[{i},{j}]").Start = 1 if i == j else 0

    m.addConstrs((x.sum("*", j) == 1 for j in range(num)))
    for i in range(N):
        m.addSOS(GRB.SOS_TYPE1, [x[i, j] for j in range(num)])
    for j in range(num):
        m.addSOS(GRB.SOS_TYPE1, [x[i, j] for i in range(N)])

    for k in range(1, num + 1):
        m.addConstrs(
            (
                cos_theta[k]
                >= dis[i][j]
                - (
                    2
                    - gp.LinExpr([1 for _ in range(k)], [x[i, l] for l in range(k)])
                    - gp.LinExpr([1 for _ in range(k)], [x[j, l] for l in range(k)])
                )
                * M
                for i, j in combinations(range(N), 2)
            ),
            name="c",
        )
        m.addConstrs(
            (
                cos_theta[k]
                >= dis[i][j]
                - (1 - gp.LinExpr([1 for _ in range(k)], [x[i, l] for l in range(k)]))
                * M
                for i, j in product(range(N), range(N, N + K))
            ),
            name="d",
        )
    m.addConstrs(cos_theta[k] <= cos_theta[k + 1] for k in range(1, num))

    m.setObjective(
        gp.LinExpr(
            [i for i in range(K + 1, K + num + 1)],
            [cos_theta[i] for i in range(1, num + 1)],
        ),
        GRB.MINIMIZE,
    )

    m.optimize()

    l = [None for _ in range(num)]
    for i, j in product(range(N), range(num)):
        if np.abs(m.getVarByName(f"x[{i},{j}]").X - 1) < eps:
            l[j] = points[i]

    return l


def incremental_sorting_single_shell(
    points: np.ndarray,
    points_per_split: np.ndarray,
    time_limit=600,
    output_flag: int = 1,
):
    """Incremental algorithm for sorting points. Given a way to split the points into segments, this function then sort them segment by segment.

    Args:
        points (np.ndarray): Points to sort.
        points_per_split (np.ndarray): Way to split number of points into segment
        time_limit (int, optional): Time limit for GUROBI to run. Defaults to 600.
        output_flag (int, optional): GUROBI output flag. Defaults to 1.

    Returns:
        Array: Sorted points
    """
    print(time_limit, type(time_limit))
    if isinstance(time_limit, float):
        time_limit = [time_limit for _ in range(len(points_per_split))]
    result = []
    picked_points = []
    remained_points = points
    flag = 0
    eps = 1e-6

    for num, t in zip(points_per_split, time_limit):
        remained_points = greedy_sorting([remained_points], start=picked_points)[0]
        pick = np.array(
            incremental_sorting_single_shell_init(
                remained_points, num, True, time_limit=t, output_flag=output_flag
            )
            if flag == 0
            else incremental_sorting_single_shell_incre(
                picked_points,
                remained_points,
                num,
                True,
                time_limit=t,
                output_flag=output_flag,
            )
        )
        flag = 1
        exclude = []
        for l in remained_points:
            if not any(all(np.abs(l - k) < eps) for k in pick):
                exclude.append(l)
        picked_points = (
            np.concatenate([picked_points, pick]) if len(picked_points) > 0 else pick
        )
        remained_points = np.array(exclude)
        result.append(pick)

    return np.concatenate(result)


def incremental_sorting_multi_shell_incre(
    fixed_points: np.ndarray,
    fixed_bval: np.ndarray,
    incre_points: np.ndarray,
    incre_bval: np.ndarray,
    fraction: List[float],
    bvalList: List[float],
    num: int,
    S: int,
    w: float = 0.5,
    start=True,
    time_limit: float = 600,
    output_flag: int = 1,
):
    """Incremental algorithm for sorting multiple shell schemes. This function sorts the next num points from incre_points with fixed_points already sorted

    Parameters
    ----------
    fixed_points : np.ndarray
        B-vectors of points that are already sorted.
    fixed_bval : np.ndarray
        B-values of points that are already sorted.
    incre_points : np.ndarray
        B-vectors of points to sort.
    incre_bval : np.ndarray
        B-values of points to sort.
    fraction : List[float]
        A balence factor for each shell. This is normally set as N_s / N for each shell with N_s points.
    bvalList : List[float]
        List of distict bvals
    num : int
        Number of points to sort
    S : int
        Number of shells
    w : float, optional
        Balance for indivial shell and combined shell, by default 0.5
    start : bool, optional
        Whether to use existing order as initialization, by default True
    time_limit : float, optional
        Time limit for GUROBI to run, by default 600
    output_flag : int, optional
        GUROBI output flag, by default 1

    Returns
    -------
    Array: Array shaped (num, 3), sorted num points
    """
    N = len(incre_points)
    K = len(fixed_points)
    M = np.pi / 2
    eps = 1e-5

    points = (
        np.concatenate([incre_points, fixed_points])
        if len(fixed_points) > 0
        else incre_points
    )
    pos2bval = {}
    for i in range(N):
        pos2bval[i] = incre_bval[i]
    for i in range(K):
        pos2bval[i + N] = fixed_bval[i]
    dis = np.clip(np.abs(points @ points.T), -1, 1)
    fixed_angle = np.cos(covering_radius(fixed_points)) if len(fixed_points) > 0 else 0
    lb = [
        max(fixed_angle, np.cos(covering_radius_upper_bound(n)))
        for n in range(K + 1, K + num + 1)
    ]

    m = gp.Model("optimalSortingSingleShell")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    x = m.addVars(N, num, lb=0, ub=1, vtype=GRB.BINARY, name="x")
    cos_theta_0 = m.addVars(
        [i for i in range(1, num + 1)],
        lb=lb,
        ub=1,
        vtype=GRB.CONTINUOUS,
        name="theta_0",
    )
    cos_theta_s = m.addVars(
        [s for s in range(S)],
        [i for i in range(1, num + 1)],
        lb=0,
        ub=1,
        vtype=GRB.CONTINUOUS,
        name="theta_s",
    )

    m.update()
    if start:
        for i, j in product(range(N), range(num)):
            m.getVarByName(f"x[{i},{j}]").Start = 1 if i == j else 0

    m.addConstrs((x.sum("*", j) == 1 for j in range(num)))
    for i in range(N):
        m.addSOS(GRB.SOS_TYPE1, [x[i, j] for j in range(num)])
    for j in range(num):
        m.addSOS(GRB.SOS_TYPE1, [x[i, j] for i in range(N)])

    for k in range(1, num + 1):
        m.addConstrs(
            (
                cos_theta_0[k]
                >= dis[i][j]
                - (
                    2
                    - gp.LinExpr([1 for _ in range(k)], [x[i, l] for l in range(k)])
                    - gp.LinExpr([1 for _ in range(k)], [x[j, l] for l in range(k)])
                )
                * M
                for i, j in combinations(range(N), 2)
            ),
            name="c",
        )
        m.addConstrs(
            (
                cos_theta_s[s, k]
                >= dis[i][j]
                - (
                    2
                    - gp.LinExpr([1 for _ in range(k)], [x[i, l] for l in range(k)])
                    - gp.LinExpr([1 for _ in range(k)], [x[j, l] for l in range(k)])
                )
                * M
                for s in range(S)
                for i, j in filter(
                    lambda x: pos2bval[x[0]] == pos2bval[x[1]] == s,
                    combinations(range(N), 2),
                )
            ),
            name="c",
        )
        m.addConstrs(
            (
                cos_theta_0[k]
                >= dis[i][j]
                - (1 - gp.LinExpr([1 for _ in range(k)], [x[i, l] for l in range(k)]))
                * M
                for i, j in product(range(N), range(N, N + K))
            ),
            name="d",
        )
        m.addConstrs(
            (
                cos_theta_s[s, k]
                >= dis[i][j]
                - (1 - gp.LinExpr([1 for _ in range(k)], [x[i, l] for l in range(k)]))
                * M
                for s in range(S)
                for i, j in filter(
                    lambda x: pos2bval[x[0]] == pos2bval[x[1]] == bvalList[s],
                    product(range(N), range(N, N + K)),
                )
            ),
            name="d",
        )
    m.addConstrs(cos_theta_0[k] <= cos_theta_0[k + 1] for k in range(1, num))
    m.addConstrs(
        cos_theta_s[s, k] <= cos_theta_s[s, k + 1]
        for s in range(S)
        for k in range(1, num)
    )

    m.setObjective(
        w
        / S
        * gp.LinExpr(
            [i * fraction[s] for s, i in product(range(S), range(K + 1, K + num + 1))],
            [cos_theta_s[s, i] for s, i in product(range(S), range(1, num + 1))],
        )
        + (1 - w)
        * gp.LinExpr(
            [i for i in range(K + 1, K + num + 1)],
            [cos_theta_0[i] for i in range(1, num + 1)],
        ),
        GRB.MINIMIZE,
    )

    m.optimize()

    l = [-1 for _ in range(num)]
    for i, j in product(range(N), range(num)):
        if np.abs(m.getVarByName(f"x[{i},{j}]").X - 1) < eps:
            l[j] = i

    return l


def incremental_sorting_multi_shell(
    vects: List[np.ndarray],
    bvalues: List[float],
    points_per_split: List[int],
    w=0.5,
    time_limit: float = 600,
    output_flag=1,
):
    """Incremental algorithm for sorting multiple shell schemes. Given a way to split the points into segments, this function then sort them segment by segment.

    Args:
        vects (List[np.ndarray]): Points to sort
        bvalues (List[float]): B-values of each point
        points_per_split (List[int]): Way to split number of points into segment
        w (float, optional): Balance for indivial shell and combined shell, by default 0.5
        time_limit (int, optional): Time limit for GUROBI to run. Defaults to 600.
        output_flag (int, optional): GUROBI output flag. Defaults to 1.

    Returns:
        (list[np.ndarray], list[int]): Sorted points and their corresponding b-value.
    """
    Ns = [len(l) for l in vects]
    N = sum(Ns)
    S = len(Ns)
    points = np.concatenate(vects)
    bvals = []
    for i, n in zip(bvalues, Ns):
        for _ in range(n):
            bvals.append(i)
    fraction = [n / N for n in Ns]

    fixed = []
    fixed_bval = []
    incre = [x for x in points]
    incre_bval = bvals

    for p in points_per_split:
        picked = incremental_sorting_multi_shell_incre(
            np.array(fixed),
            fixed_bval,
            np.array(incre),
            incre_bval,
            fraction,
            bvalues,
            p,
            S,
            w=w,
            time_limit=time_limit,
            output_flag=output_flag,
        )
        for i in picked:
            fixed.append(incre[i])
            fixed_bval.append(incre_bval[i])
        for i in sorted(picked, reverse=True):
            incre.pop(i)
            incre_bval.pop(i)

    return fixed, fixed_bval
