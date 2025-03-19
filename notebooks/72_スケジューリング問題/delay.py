# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pyscipopt==5.4.1",
# ]
# ///

import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import random
    import pyscipopt as scip
    return random, scip


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 1 機械総納期遅れ最小化問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 離接定式化""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - 機械: 1 つだけ
        - ジョブ: $J = \{ 1, \dots, n \}$
        - $p_j$: ジョブ $j$ の処理時間
        - $d_j$: ジョブ $j$ の納期

        各ジョブ $j$ について $d_j$ からの遅れの重み付き和を最小化する.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $M$ を大きな定数として

        \begin{align*}
        &\text{minimize} &\sum_{j \in J} w_j T_j \\
        &\text{s.t.} &x_{jk} + x_{kj} &= 1 \ &(\forall j < k) \\
        & &x_{jk} + x_{kl} + x_{lj} &\leq 2 \ &(\forall j \neq k \neq l) \\
        & &\sum_{k \neq j} p_k x_{kj} +p_j &\leq d_j + T_j  \ &(\forall j \in J) \\
        & &x_{jk} &\in \{ 0, 1 \}
        \end{align*}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 実装""")
    return


@app.cell
def _(random):
    def make_data(n):
        """
        Data generator for the one machine scheduling problem.
        """
        p, r, d, w = {}, {}, {}, {}

        J = range(1, n + 1)

        for j in J:
            p[j] = random.randint(1, 4)
            w[j] = random.randint(1, 3)

        T = sum(p)
        for j in J:
            r[j] = random.randint(0, 5)
            d[j] = r[j] + random.randint(0, 5)

        return J, p, r, d, w
    return (make_data,)


@app.cell
def _(scip):
    def scheduling_linear_ordering(J, p, d, w):
        """
        scheduling_linear_ordering: model for the one machine total weighted tardiness problem

        Model for the one machine total weighted tardiness problem
        using the linear ordering formulation

        Parameters:
            - J: set of jobs
            - p[j]: processing time of job j
            - d[j]: latest non-tardy time for job j
            - w[j]: weighted of job j,  the objective is the sum of the weighted completion time

        Returns a model, ready to be solved.
        """
        model = scip.Model("scheduling: linear ordering")

        T, x = {}, {}  # tardiness variable,  x[j,k] =1 if job j precedes job k, =0 otherwise
        for j in J:
            T[j] = model.addVar(vtype="C", name="T(%s)" % (j))
            for k in J:
                if j != k:
                    x[j, k] = model.addVar(vtype="B", name="x(%s,%s)" % (j, k))

        for j in J:
            model.addCons(
                scip.quicksum(p[k] * x[k, j] for k in J if k != j) - T[j] <= d[j] - p[j],
                "Tardiness(%r)" % (j),
            )

            for k in J:
                if k <= j:
                    continue
                model.addCons(x[j, k] + x[k, j] == 1, "Disjunctive(%s,%s)" % (j, k))

                for ell in J:
                    if ell == j or ell == k:
                        continue
                    # if ell > k:
                        # j < k < ell
                        # 1 -> 3 -> 2
                        # x[1, 3], x[3, 2], x[2, 1]
                    model.addCons(
                        x[j, k] + x[k, ell] + x[ell, j] <= 2,
                        "Triangle(%s,%s,%s)" % (j, k, ell),
                    )

        model.setObjective(scip.quicksum(w[j] * T[j] for j in J), sense="minimize")

        return model, x, T
    return (scheduling_linear_ordering,)


@app.cell
def _(make_data, scheduling_linear_ordering):
    n = 5  # number of jobs
    J, p, r, d, w = make_data(n)

    model, x, T = scheduling_linear_ordering(J, p, d, w)
    model.optimize()
    z = model.getObjVal()
    for (i, j) in x:
        if model.getVal(x[i, j]) > 0.5:
            print("x(%s) = %s" % ((i, j), int(model.getVal(x[i, j]) + 0.5)))
    for i in T:
        print("T(%s) = %s" % (i, int(model.getVal(T[i]) + 0.5)))
    print("Opt.value by the linear ordering formulation=", z)
    return J, T, d, i, j, model, n, p, r, w, x, z


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
