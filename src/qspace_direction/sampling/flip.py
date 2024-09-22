from itertools import combinations, product
from typing import List

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def dirflip(
    points: np.ndarray,
    time_limit=600,
    output_flag=1,
):
    N = len(points)
    dis = np.arccos(np.clip(points @ points.T, -1, 1))

    m = gp.Model("dirflip")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    h = m.addVars(N, vtype=GRB.BINARY, name="h")
    x = m.addVars(N, N, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
    theta = m.addVar(lb=0, name="theta")

    m.addConstrs(
        (x[i, j] <= h[i] + h[j] for i, j in combinations(range(N), 2)),
    )
    m.addConstrs(
        (x[i, j] >= h[i] - h[j] for i, j in combinations(range(N), 2)),
    )
    m.addConstrs(
        (x[i, j] >= h[j] - h[i] for i, j in combinations(range(N), 2)),
    )
    m.addConstrs(
        (x[i, j] <= 2 - h[i] - h[j] for i, j in combinations(range(N), 2)),
    )
    m.addConstrs(
        (
            (1 - x[i, j]) * dis[i][j] + x[i, j] * (np.pi - dis[i][j]) >= theta
            for i, j in combinations(range(N), 2)
        ),
    )

    m.setObjective(theta, GRB.MAXIMIZE)

    m.optimize()

    l = []
    for i in range(N):
        if m.getVarByName(f"h[{i}]").X == 1:
            l.append(-points[i])
        else:
            l.append(points[i])

    return l


def dirflipEEM(
    points: np.ndarray,
    order=1,
    time_limit=600,
    output_flag=1,
):
    N = len(points)
    dis = np.clip(points @ points.T, -1, 1)
    f = lambda x: 1 / ((1 - x) ** order)

    m = gp.Model("dirflipEEM")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    h = m.addVars(N, vtype=GRB.BINARY, name="h")
    x = m.addVars(N, N, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")

    m.addConstrs(
        (x[i, j] <= h[i] + h[j] for i, j in combinations(range(N), 2)),
    )
    m.addConstrs(
        (x[i, j] >= h[i] - h[j] for i, j in combinations(range(N), 2)),
    )
    m.addConstrs(
        (x[i, j] >= h[j] - h[i] for i, j in combinations(range(N), 2)),
    )
    m.addConstrs(
        (x[i, j] <= 2 - h[i] - h[j] for i, j in combinations(range(N), 2)),
    )

    m.setObjective(
        gp.quicksum(
            [
                (1 - x[i, j]) * f(dis[i, j]) + x[i, j] * f(-dis[i, j])
                for i, j in combinations(range(N), 2)
            ]
        ),
        GRB.MINIMIZE,
    )

    m.optimize()

    l = []
    for i in range(N):
        if m.getVarByName(f"h[{i}]").X == 1:
            l.append(-points[i])
        else:
            l.append(points[i])

    return l


def dirflipMultiShell(
    points: List[np.ndarray],
    w=0.5,
    time_limit=600,
    output_flag=1,
):
    N_s = [len(l) for l in points]
    S = len(points)

    consPoints = np.concatenate(points)
    dis = np.arccos(np.clip(consPoints @ consPoints.T, -1, 1))
    indices = np.cumsum(N_s).tolist()
    indices.insert(0, 0)

    m = gp.Model("dirflipMultiShell")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    h = m.addVars(
        [(i, s) for s in range(S) for i in range(N_s[s])], vtype=GRB.BINARY, name="h"
    )
    x = m.addVars(
        [
            (i, s, j, t)
            for s in range(S)
            for i in range(N_s[s])
            for t in range(S)
            for j in range(N_s[t])
        ],
        lb=0,
        ub=1,
        vtype=GRB.CONTINUOUS,
        name="x",
    )
    theta_s = m.addVars(
        S,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="theta_s",
    )
    theta_0 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta_0")

    m.addConstrs(
        (
            x[i, s, j, t] <= h[i, s] + h[j, t]
            for s in range(S)
            for i in range(N_s[s])
            for t in range(S)
            for j in range(N_s[t])
        ),
    )
    m.addConstrs(
        (
            x[i, s, j, t] >= h[i, s] - h[j, t]
            for s in range(S)
            for i in range(N_s[s])
            for t in range(S)
            for j in range(N_s[t])
        ),
    )
    m.addConstrs(
        (
            x[i, s, j, t] >= h[j, t] - h[i, s]
            for s in range(S)
            for i in range(N_s[s])
            for t in range(S)
            for j in range(N_s[t])
        ),
    )
    m.addConstrs(
        (
            x[i, s, j, t] <= 2 - h[i, s] - h[j, t]
            for s in range(S)
            for i in range(N_s[s])
            for t in range(S)
            for j in range(N_s[t])
        ),
    )
    for s in range(S):
        m.addConstrs(
            (
                (1 - x[i, s, j, s]) * dis[indices[s] + i][indices[s] + j]
                + x[i, s, j, s] * (np.pi - dis[indices[s] + i][indices[s] + j])
                >= theta_s[s]
                for i, j in combinations(range(N_s[s]), 2)
            ),
        )
    for s, t in combinations(range(S), 2):
        m.addConstrs(
            (
                (1 - x[i, s, j, t]) * dis[indices[s] + i][indices[t] + j]
                + x[i, s, j, t] * (np.pi - dis[indices[s] + i][indices[t] + j])
                >= theta_0
                for i, j in product(range(N_s[s]), range(N_s[t]))
            ),
        )

    m.setObjective(w / S * theta_s.sum() + (1 - w) * theta_0, GRB.MAXIMIZE)

    m.optimize()

    l = [[] for _ in range(S)]
    for s in range(S):
        for i in range(N_s[s]):
            if m.getVarByName(f"h[{i},{s}]").X == 1:
                l[s].append(-points[s][i])
            else:
                l[s].append(points[s][i])

    return l


def dirflipMultiShellEEM(
    points: List[np.ndarray],
    w=0.5,
    order=1,
    time_limit=600,
    output_flag=1,
):
    N_s = [len(l) for l in points]
    S = len(points)

    consPoints = np.concatenate(points)
    L = len(consPoints)
    dis = np.clip(consPoints @ consPoints.T, -1, 1)
    f = lambda x: 1 / ((1 - x) ** order)
    indices = np.cumsum(N_s).tolist()
    indices.insert(0, 0)

    m = gp.Model("dirflipMultiShell")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    h = m.addVars(
        [(i, s) for s in range(S) for i in range(N_s[s])], vtype=GRB.BINARY, name="h"
    )
    x = m.addVars(
        [
            (i, s, j, t)
            for s in range(S)
            for i in range(N_s[s])
            for t in range(S)
            for j in range(N_s[t])
        ],
        lb=0,
        ub=1,
        vtype=GRB.CONTINUOUS,
        name="x",
    )

    m.addConstrs(
        (
            x[i, s, j, t] <= h[i, s] + h[j, t]
            for s in range(S)
            for i in range(N_s[s])
            for t in range(S)
            for j in range(N_s[t])
        ),
    )
    m.addConstrs(
        (
            x[i, s, j, t] >= h[i, s] - h[j, t]
            for s in range(S)
            for i in range(N_s[s])
            for t in range(S)
            for j in range(N_s[t])
        ),
    )
    m.addConstrs(
        (
            x[i, s, j, t] >= h[j, t] - h[i, s]
            for s in range(S)
            for i in range(N_s[s])
            for t in range(S)
            for j in range(N_s[t])
        ),
    )
    m.addConstrs(
        (
            x[i, s, j, t] <= 2 - h[i, s] - h[j, t]
            for s in range(S)
            for i in range(N_s[s])
            for t in range(S)
            for j in range(N_s[t])
        ),
    )

    m.setObjective(
        w
        / S
        * gp.quicksum(
            [
                (
                    (1 - x[i, s, j, s]) * f(dis[indices[s] + i][indices[s] + j])
                    + x[i, s, j, s] * f(-dis[indices[s] + i][indices[s] + j])
                )
                / (N_s[s] * N_s[s])
                for s in range(S)
                for i, j in combinations(range(N_s[s]), 2)
            ],
        )
        + (1 - w)
        / (L * L)
        * gp.quicksum(
            [
                (1 - x[i, s, j, t]) * f(dis[indices[s] + i][indices[t] + j])
                + x[i, s, j, t] * f(-dis[indices[s] + i][indices[t] + j])
                for s, t in combinations(range(S), 2)
                for i, j in product(range(N_s[s]), range(N_s[t]))
            ]
        ),
        GRB.MINIMIZE,
    )

    m.optimize()

    l = [[] for _ in range(S)]
    for s in range(S):
        for i in range(N_s[s]):
            if m.getVarByName(f"h[{i},{s}]").X == 1:
                l[s].append(-points[s][i])
            else:
                l[s].append(points[s][i])

    return l
