# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "ortools==9.12.4544",
#     "pycsp3==2.4.3",
#     "pyscipopt==5.4.1",
# ]
# ///

import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import random
    import pyscipopt as scip
    from ortools.sat.python import cp_model
    return cp_model, random, scip


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 1 機械リリース時刻付き重み付き完了時刻和最小化問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - 機械: 1 つだけ
        - ジョブ $J = \{ 1, \dots, n \}$
        - 各ジョブの処理時間: $p_j \ (\forall j \in J)$
        - 各ジョブの重要度: $w_j \ (\forall j \in J)$
        - 各ジョブのリリース時刻: $r_j \ (\forall j \in J)$
        - 各ジョブの処理完了時刻: $C_j \ (\forall j \in J)$

        $C_j$ の重み付き和を最小化する.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 離接定式化(Disjunctive formulation)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $M$ を大きな定数として

        \begin{align*}
        &\text{minimize} &\sum_{j=1}^n w_j s_j + \sum_{j=1}^n w_j p_j \\
        &\text{s.t.} &s_j + p_j - M (1 - x_{jk}) &\leq s_k \ &(\forall j \neq k) \\
        & &x_{jk} + x_{kj} &= 1 \ &(\forall j < k) \\
        & &s_j &\geq r_j \ &(\forall j \in J) \\
        & &x_{jk} &\in \{0, 1\} \ &(\forall j \neq k)
        \end{align*}

        - 決定変数
            - $s_j$: ジョブ $j$ の開始時刻
            - $x_{jk}$: ジョブ $j$ がジョブ $k$ に先行するとき $1$
        - 補足
            - 目的関数の第 2 項目は定数であるため第 1 項だけを最小化すればよい
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

        random.seed(0)
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
    class ModelDisjunctiveScip:
        def __init__(self, J, p, r, w):
            self.J = J
            self.p = p
            self.r = r
            self.w = w

            self.scip = scip.Model("scheduling: disjunctive")

            # Big M
            M = max(self.r.values()) + sum(self.p.values())

            # start time variable, x[j,k] = 1 if job j precedes job k, 0 otherwise
            self.x = {}
            self.s = {}

            ub = sum(self.p[j] for j in self.J)
            for j in self.J:
                self.s[j] = self.scip.addVar(lb=r[j], ub=ub, vtype="C", name=f"s[{j}]")
                for k in self.J:
                    if j != k:
                        self.x[j, k] = self.scip.addVar(vtype="B", name=f"x[{j},{k}]")

            for j in self.J:
                for k in self.J:
                    if j != k:
                        self.scip.addCons(
                            self.s[j] - self.s[k] + M * self.x[j, k] <= (M - self.p[j]), f"Bound[{j},{k}]"
                        )

                    if j < k:
                        self.scip.addCons(self.x[j, k] + self.x[k, j] == 1, f"Disjunctive[{j},{k}]")

            self.scip.setObjective(scip.quicksum(self.w[_j] * self.s[_j] for _j in self.J), sense="minimize")

        def solve(self) -> None:
            self.scip.optimize()

        def get_z(self):
            return self.scip.getObjVal() + sum([self.w[_j] * self.p[_j] for _j in self.J])

        def get_seq(self):
            return [_j for (_t, _j) in sorted([(int(self.scip.getVal(self.s[_j]) + 0.5), _j) for _j in self.s])]
    return (ModelDisjunctiveScip,)


@app.cell
def _(scip):
    class _MyInterval:
        # [start, end)
        def __init__(self, scip: scip.Model, lb: int, ub: int, size: int):
            self.lb = lb
            self.ub = ub
            self.size = size
            self.start = scip.addVar(lb=self.lb, ub=self.ub - self.size, vtype='C')
            self.end = self.start + self.size

    def _my_add_no_overlap(scip: scip.Model, jobs: dict[int, _MyInterval]):
        if len(jobs) == 0:
            return
        for _j1 in jobs.keys():
            for _j2 in jobs.keys():
                if _j2 <= _j1:
                    continue
                big_m = max(jobs[_j1].ub - jobs[_j2].lb, jobs[_j2].ub - jobs[_j1].lb)
                tmp1 = scip.addVar(vtype='B')
                scip.addCons(
                    jobs[_j1].end <= jobs[_j2].start + big_m * (1 - tmp1)
                )
                tmp2 = scip.addVar(vtype='B')
                scip.addCons(
                    jobs[_j2].end <= jobs[_j1].start + big_m * (1 - tmp2)
                )
                scip.addCons(tmp1 + tmp2 >= 1)

    class ModelIntervalScip:
        def __init__(self, J, p, r, w):
            self.J = J
            self.p = p
            self.r = r
            self.w = w

            self.scip = scip.Model("scheduling: disjunctive")

            ub = sum(p[_j] for _j in J)
            self.jobs = {_j: _MyInterval(self.scip, self.r[_j], ub, self.p[_j]) for _j in self.J}
            _my_add_no_overlap(self.scip, self.jobs)

            self.scip.setObjective(scip.quicksum(self.w[_j] * self.jobs[_j].start for _j in self.J), sense="minimize")

        def solve(self) -> None:
                self.scip.optimize()

        def get_z(self):
            return self.scip.getObjVal() + sum([self.w[_j] * self.p[_j] for _j in self.J])

        def get_seq(self):
            return [_j for (_t, _j) in sorted([(int(self.scip.getVal(self.jobs[_j].start) + 0.5), _j) for _j in self.jobs])]
    return (ModelIntervalScip,)


@app.cell
def _(cp_model):
    class ModelDisjunctiveCpSat:
        def __init__(self, J, p, r, w):
            self.J = J
            self.p = p
            self.r = r
            self.w = w

            self.cp = cp_model.CpModel()
            self.solver = cp_model.CpSolver()

            # Big M
            M = max(r.values()) + sum(p.values())

            # start time variable, x[j,k] = 1 if job j precedes job k, 0 otherwise
            self.x = {}
            self.s = {}

            ub = sum(p[j] for j in J)
            for j in J:
                self.s[j] = self.cp.new_int_var(lb=r[j], ub=ub, name=f"s[{j}]")
                for k in J:
                    if j != k:
                        self.x[j, k] = self.cp.new_bool_var(name=f"x[{j},{k}]")

            for j in J:
                for k in J:
                    if j != k:
                        self.cp.add(self.s[j] - self.s[k] + M * self.x[j, k] <= (M - p[j]))

                    if j < k:
                        self.cp.add(self.x[j, k] + self.x[k, j] == 1)

            self.cp.minimize(sum(self.w[j] * self.s[j] for j in J))

        def solve(self) -> None:
            self.solver.parameters.log_search_progress = True
            self.solver.solve(self.cp)

        def get_z(self):
            return self.solver.objective_value + sum([self.w[_j] * self.p[_j] for _j in self.J])

        def get_seq(self):
            return [_j for (_t, _j) in sorted([(int(self.solver.value(self.s[_j]) + 0.5), _j) for _j in self.s])]
    return (ModelDisjunctiveCpSat,)


@app.cell
def _(cp_model):
    class ModelIntervalCpSat:
        def __init__(self, J, p, r, w):
            self.J = J
            self.p = p
            self.r = r
            self.w = w

            self.cp = cp_model.CpModel()
            self.solver = cp_model.CpSolver()

            ub = sum(p[_j] for _j in J)
            s = {_j: self.cp.new_int_var(lb=self.r[_j], ub=ub, name=f"s[{_j}]") for _j in self.J}
            self.jobs = {_j: self.cp.new_fixed_size_interval_var(s[_j], self.p[_j], f"jobs[{_j}]") for _j in self.J}
            self.cp.add_no_overlap(self.jobs.values())

            self.cp.minimize(sum(self.w[_j] * self.jobs[_j].start_expr() for _j in J))

        def solve(self) -> None:
            self.solver.parameters.log_search_progress = True
            self.solver.solve(self.cp)

        def get_z(self):
            return self.solver.objective_value + sum([self.w[_j] * self.p[_j] for _j in self.J])

        def get_seq(self):
            return [_j for (_t, _j) in sorted([(self.solver.value(self.jobs[_j].start_expr()), _j) for _j in self.jobs])]
    return (ModelIntervalCpSat,)


@app.cell
def _(make_data):
    n = 30
    J, p, r, d, w = make_data(n)
    return J, d, n, p, r, w


@app.cell(hide_code=True)
def _(mo):
    run_scip_disj = mo.ui.run_button(label="Run", full_width=True)
    run_scip_disj
    return (run_scip_disj,)


@app.cell
def _(J, ModelDisjunctiveScip, mo, p, r, run_scip_disj, w):
    if run_scip_disj.value:
        _model = ModelDisjunctiveScip(J, p, r, w)
        with mo.redirect_stderr():
            _model.solve()

        _z = _model.get_z()
        _seq = _model.get_seq()
        mo.md(f"""
        - Opt.value by Disjunctive Formulation: {_z}
        - Solution: {_seq}
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_scip_interval = mo.ui.run_button(label="Run", full_width=True)
    run_scip_interval
    return (run_scip_interval,)


@app.cell
def _(J, ModelIntervalScip, mo, p, r, run_scip_interval, w):
    if run_scip_interval.value:
        _model = ModelIntervalScip(J, p, r, w)
        with mo.redirect_stderr():
            _model.solve()

        _z = _model.get_z()
        _seq = _model.get_seq()
        mo.md(f"""
        - Optimal value: {_z}
        - Solution: {_seq}
        """)
    return


@app.cell
def _(J, ModelDisjunctiveCpSat, mo, p, r, w):
    _model = ModelDisjunctiveCpSat(J, p, r, w)
    with mo.redirect_stderr():
        _model.solve()

    _z = _model.get_z()
    _seq = _model.get_seq()
    mo.md(f"""
    - Opt.value by Disjunctive Formulation: {_z}
    - Solution: {_seq}
    """)
    return


@app.cell
def _(J, ModelIntervalCpSat, mo, p, r, w):
    _model = ModelIntervalCpSat(J, p, r, w)
    with mo.redirect_stderr():
        _model.solve()

    _z = _model.get_z()
    _seq = _model.get_seq()
    mo.md(f"""
    - Optimal value: {_z}
    - Solution: {_seq}
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
