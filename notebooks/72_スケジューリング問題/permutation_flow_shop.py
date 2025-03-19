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
    mo.md(r"""# 順列フローショップ問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $n$ 個のジョブを $m$ 台のマシンで順番に処理する.
        各ジョブはマシン 1, マシン 2, ... で順に処理されマシン $m$ で処理されると完了になる.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="https://www.researchgate.net/profile/Mariusz-Makuchowski/publication/280775329/figure/fig1/AS:284468087672848@1444833885900/Schedules-of-the-different-variants-of-the-flow-shop-problem.png",
        width=800,
        caption="https://www.researchgate.net/figure/Schedules-of-the-different-variants-of-the-flow-shop-problem_fig1_280775329"
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 位置データ定式化""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 定数

        - ジョブ: $J = \{ 1, \dots, n \}$
        - マシン: $M = \{ 1, \dots, m \}$
        - 処理時間: $p_{ij} \ (\forall i \in M, \forall j \in J)$

        ## 決定変数

        - $x_{j \kappa} \in \{ 0, 1 \}$: ジョブ $j$ を並べた時の順番が $\kappa$ 番目であるとき $1$.
        - $s_{i \kappa}$: マシン $i$ の $\kappa$ 番目に並べられているジョブの開始時刻
        - $f_{i \kappa}$: マシン $i$ の $\kappa$ 番目に並べられているジョブの終了時刻
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        \begin{align*}
        &\text{minimize} &f_{mn} \\
        &\text{s.t.} &\sum_{\kappa} x_{j \kappa} &= 1 \ &(\forall j \in J) \\
        & &\sum_{j \in J} x_{j \kappa} &= 1 \ &(\forall \kappa = 1, \dots, n) \\
        & &f_{i \kappa} &\leq s_{i,\kappa+1} \ &(\forall i \in M, \forall \kappa = 1, \dots, n-1) \\
        & &s_{i \kappa} + \sum_{j \in J} p_{ij} x_{j \kappa} &\leq f_{i \kappa} \ &(\forall i \in M, \forall \kappa = 1, \dots, n) \\
        & &f_{i \kappa} &\leq s_{i+1, \kappa} \ &(\forall i \in M \setminus \{m\}, \forall \kappa = 1, \dots, n) \\
        & &x_{j \kappa} &\in \{ 0, 1 \} \ &(\forall j \in J, \kappa = 1, \dots, n) \\
        & &s_{i \kappa} &\geq 0 \ &(\forall i \in M, \kappa = 1, \dots, n) \\
        & &f_{i \kappa} &\geq 0 \ &(\forall i \in M, \kappa = 1, \dots, n) \\
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
    def make_data_permutation_flow_shop(n, m):
        """make_data: prepare matrix of m times n random processing times"""
        p = {}
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                p[i, j] = random.randint(1, 10)
        return p
    return (make_data_permutation_flow_shop,)


@app.cell
def _(scip):
    def permutation_flow_shop(n, m, p):
        """ permutation_flow_shop problem
        Parameters:
            - n: number of jobs
            - m: number of machines
            - p[i,j]: processing time of job i on machine j
        Returns a model, ready to be solved.
        """
        model = scip.Model("permutation flow shop")
        x, s, f = {}, {}, {}
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                x[j, k] = model.addVar(vtype="B", name="x(%s,%s)" % (j, k))

        for i in range(1, m + 1):
            for k in range(1, n + 1):
                s[i, k] = model.addVar(vtype="C", name="start(%s,%s)" % (i, k))
                f[i, k] = model.addVar(vtype="C", name="finish(%s,%s)" % (i, k))

        for j in range(1, n + 1):
            model.addCons(
                scip.quicksum(x[j, k] for k in range(1, n + 1)) == 1, "Assign1(%s)" % (j)
            )
            model.addCons(
                scip.quicksum(x[k, j] for k in range(1, n + 1)) == 1, "Assign2(%s)" % (j)
            )

        for i in range(1, m + 1):
            for k in range(1, n + 1):
                if k != n:
                    model.addCons(f[i, k] <= s[i, k + 1], "FinishStart(%s,%s)" % (i, k))
                if i != m:
                    model.addCons(f[i, k] <= s[i + 1, k], "Machine(%s,%s)" % (i, k))

                model.addCons(
                    s[i, k] + scip.quicksum(p[i, j] * x[j, k] for j in range(1, n + 1))
                    <= f[i, k],
                    "StartFinish(%s,%s)" % (i, k),
                )

        model.setObjective(f[m, n], sense="minimize")

        return model, x, s, f
    return (permutation_flow_shop,)


@app.cell
def _(mo):
    optimize_model = mo.ui.run_button(label="Run")
    optimize_model
    return (optimize_model,)


@app.cell
def _(make_data_permutation_flow_shop, optimize_model, permutation_flow_shop):
    if optimize_model.value:
    # if True:
        n = 5
        m = 5
        p = make_data_permutation_flow_shop(n, m)

        model, x, s, f = permutation_flow_shop(n, m, p)
        model.optimize()
        print("Opt.value=", model.getObjVal())
    return f, m, model, n, p, s, x


@app.cell
def _(model):
    # if optimize_model.value:
    if True:
        model.getObjVal()
    return


if __name__ == "__main__":
    app.run()
