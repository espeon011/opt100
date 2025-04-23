# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "highspy==1.10.0",
#     "marimo",
#     "ortools==9.12.4544",
#     "pydantic==2.11.3",
# ]
# ///

import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    from typing import Self
    from pathlib import Path
    import pydantic
    import highspy
    from ortools.sat.python import cp_model
    return Path, Self, cp_model, highspy, os, pydantic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 資源制約付きプロジェクトスケジューリング問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 入力データ

        - ジョブの集合 $\mathrm{Job}$, 添字は $j, k$
        - 資源の集合 $\mathrm{Res}$, 添字は $r$
        - ジョブ間の時間成約を表す集合 $\mathrm{Prec} \subset \mathrm{Job} \times \mathrm{Job}$
            - $(j,k) \in \mathrm{Prec}$ のとき ジョブ $j$ とジョブ $k$ の時刻間に何かしらの関係がある. 
        - 最大の期数 $T$, 添字は $t, s \in \{1, \dots, T \}$
            - 期間 $t$ は時刻 $t-1$ から時刻 $t$ までであるとする. 
        - ジョブ $j$ の処理時間 $p_j$
        - ジョブ $j$ を期 $t$ に開始したときの費用 $\mathrm{Cost}_{jt}$
        - ジョブ $j$ の開始後 $t$ 期経過時の処理に要する資源 $r$ の量 $a_{jrt}$
        - 期 $t$ における資源 $r$ の使用可能量上限 $\mathrm{RUB}_{rt}$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 変数

        - $x_{jt} \in \{ 0, 1 \}$: ジョブ $j$ を期 $t$ に開始するとき $1$, それ以外は $0$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 目的関数

        $$
        \min \sum_{j \in \mathrm{Job}} \sum_{t=1}^{T-p_j+1} \mathrm{Cost}_{jt} x_{jt}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 制約条件

        - ジョブ遂行成約:
          $\sum_{t=1}^{T-p_j+1} x_{jt} = 1 \quad (\forall j \in \mathrm{Job})$
        - 資源成約:
          $\sum_{j \in \mathrm{Job}} \sum_{s = \max(t - p_j + 1, 1)}^{\min(t, T - p_j + 1)} a_{jr,t-s} x_{js} \leq \mathrm{RUB}_{rt} \quad (\forall r \in \mathrm{Res}, \forall t \in \{ 1, \dots, T \})$
        - 時間制約: $\sum_{t=2}^{T-p_j+1} (t-1)x_{jt} + p_j \leq \sum_{t=2}^{T-p_k+1} (t-1) x_{kt} \quad (\forall (j,k) \in \mathrm{Prec})$
        """
    )
    return


@app.cell
def multidict():
    # https://scmopt.github.io/manual/15mypulp.html#multidict%E9%96%A2%E6%95%B0

    def multidict(d: dict):
        ret = [list(d.keys())]
        for k, arr in d.items():
            if type(arr) is not list:
                arr = [arr]
            append_num = 1 + len(arr) - len(ret)
            if append_num > 0:
                ret = ret + [{} for _ in range(append_num)]
            for i, val in enumerate(arr):
                ret[i + 1][k] = val
        return ret
    return (multidict,)


@app.cell
def make_1r(multidict):
    def make_1r():
        J, p = multidict(
            {  # jobs, processing times
                1: 1,
                2: 3,
                3: 2,
                4: 2,
            }
        )
        P = [(1, 2), (1, 3), (2, 4)]
        R = [1]
        T = 6
        c = {}
        for j in J:
            for t in range(1, T - p[j] + 2):
                c[j, t] = 1 * (t - 1 + p[j])
        a = {
            (1, 1, 0): 2,
            (2, 1, 0): 2,
            (2, 1, 1): 1,
            (2, 1, 2): 1,
            (3, 1, 0): 1,
            (3, 1, 1): 1,
            (4, 1, 0): 1,
            (4, 1, 1): 2,
        }
        RUB = {(1, 1): 2, (1, 2): 2, (1, 3): 1, (1, 4): 2, (1, 5): 2, (1, 6): 2}
        return (J, P, R, T, p, c, a, RUB)
    return (make_1r,)


@app.cell
def make_2r(multidict):
    def make_2r():
        J, p = multidict(
            {  # jobs, processing times
                1: 2,
                2: 2,
                3: 3,
                4: 2,
                5: 5,
            }
        )
        P = [(1, 2), (1, 3), (2, 4)]
        R = [1, 2]
        T = 6
        c = {}
        for j in J:
            for t in range(1, T - p[j] + 2):
                c[j, t] = 1 * (t - 1 + p[j])
        a = {
            # resource 1:
            (1, 1, 0): 2,
            (1, 1, 1): 2,
            (2, 1, 0): 1,
            (2, 1, 1): 1,
            (3, 1, 0): 1,
            (3, 1, 1): 1,
            (3, 1, 2): 1,
            (4, 1, 0): 1,
            (4, 1, 1): 1,
            (5, 1, 0): 0,
            (5, 1, 1): 0,
            (5, 1, 2): 1,
            (5, 1, 3): 0,
            (5, 1, 4): 0,
            # resource 2:
            (1, 2, 0): 1,
            (1, 2, 1): 0,
            (2, 2, 0): 1,
            (2, 2, 1): 1,
            (3, 2, 0): 0,
            (3, 2, 1): 0,
            (3, 2, 2): 0,
            (4, 2, 0): 1,
            (4, 2, 1): 2,
            (5, 2, 0): 1,
            (5, 2, 1): 2,
            (5, 2, 2): 1,
            (5, 2, 3): 1,
            (5, 2, 4): 1,
        }
        RUB = {
            (1, 1): 2,
            (1, 2): 2,
            (1, 3): 2,
            (1, 4): 2,
            (1, 5): 2,
            (1, 6): 2,
            (1, 7): 2,
            (2, 1): 2,
            (2, 2): 2,
            (2, 3): 2,
            (2, 4): 2,
            (2, 5): 2,
            (2, 6): 2,
            (2, 7): 2,
        }
        return (J, P, R, T, p, c, a, RUB)
    return (make_2r,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## HiGHS によるモデリング""")
    return


@app.cell
def _(highspy):
    class Model1Highs:
        def __init__(self, J, P, R, T, p, c, a, RUB):
            self.model = highspy.Highs()

            # s - start time variable
            # x=1 if job j starts on period t
            self.s, self.x = {}, {}
            for j in J:
                self.s[j] = self.model.addVariable()
                for t in range(1, T - p[j] + 2):
                    self.x[j, t] = self.model.addBinary()

            for j in J:
                # job execution constraints
                self.model.addConstr(
                    sum(self.x[j, t] for t in range(1, T - p[j] + 2)) == 1
                )

                # start time constraints
                self.model.addConstr(
                    sum((t - 1) * self.x[j, t] for t in range(2, T - p[j] + 2)) == self.s[j]
                )

            # resource upper bound constraints
            for t in range(1, T+1):
                for r in R:
                    self.model.addConstr(
                        sum(
                            a[j, r, t - t_] * self.x[j, t_]
                            for j in J
                            for t_ in range(max(t - p[j] + 1, 1), min(t + 1, T - p[j] + 2))
                        )
                        <= RUB[r, t]
                    )

            # time (precedence) constraints, i.e., s[k]-s[j] >= p[j]
            for (j, k) in P:
                self.model.addConstr(self.s[k] - self.s[j] >= p[j])

            self.objective = sum(c[j, t] * self.x[j, t] for (j, t) in self.x)
            # self.model.minimize(self.objective)

        def solve(self) -> None:
            # self.model.run()
            self.model.minimize(self.objective)
            self.solution = self.model.getSolution()
            self.info = self.model.getInfo()
    return (Model1Highs,)


@app.cell
def _(Model1Highs, make_1r):
    (J1, P1, R1, T1, p1, c1, a1, RUB1) = make_1r()
    model1 = Model1Highs(J1, P1, R1, T1, p1, c1, a1, RUB1)

    model1.solve()
    return J1, P1, R1, RUB1, T1, a1, c1, model1, p1


@app.cell
def _(model1):
    print (f"Opt.value = {model1.info.objective_function_value}")

    for (_j, _t) in model1.x:
        _val = model1.solution.col_value[model1.x[_j, _t].index]
        if _val > 0.5:
            print(f"x[{_j},{_t}] = {_val}")

    for _j in model1.s:
        _val = model1.solution.col_value[model1.s[_j].index]
        print(f"s[{_j}] = {_val}")
    return


@app.cell
def _(Model1Highs, make_2r):
    (J2, P2, R2, T2, p2, c2, a2, RUB2) = make_2r()
    model2 = Model1Highs(J2, P2, R2, T2, p2, c2, a2, RUB2)

    model2.solve()
    return J2, P2, R2, RUB2, T2, a2, c2, model2, p2


@app.cell
def _(model2):
    print (f"Opt.value = {model2.info.objective_function_value}")

    for (_j, _t) in model2.x:
        _val = model2.solution.col_value[model2.x[_j, _t].index]
        if _val > 0.5:
            print(f"x[{_j},{_t}] = {_val}")

    for _j in model2.s:
        _val = model2.solution.col_value[model2.s[_j].index]
        print(f"s[{_j}] = {_val}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## インスタンス""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### kobe-scheduling

        - リンク: https://github.com/ptal/kobe-scheduling
        - データフォーマットはデータによって異なり, 特に説明もない. 
          https://github.com/ptal/kobe-scheduling/data/rcpsp/patterson.rcp であれば (多分)
            - 1 行目: ジョブの数, リソースの種類
            - 3 行目: 各リソースの上限値
            - 5 行目以降: ジョブの情報が並ぶ. 
              処理時間, [リソースの消費数, ...], 後続ジョブ数, [後続ジョブ番号, ...]
        - 目的関数は makespan (多分)
        """
    )
    return


@app.cell
def _(Path, __file__, os):
    parent = str(Path(os.path.abspath(__file__)).parent)
    data_dir = Path(parent, "kobe-scheduling", "data", "rcpsp", "patterson.rcp")
    return data_dir, parent


@app.cell
def _(Self, pydantic):
    class Job(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(frozen=True)
        id: int
        time: int
        res_usages: list[int]

    class Resource(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(frozen=True)
        ub: int

    class Condition(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(frozen=True)
        jobs: list[Job]
        ress: list[Resource]
        prec: set[tuple[int,int]]

        @staticmethod
        def from_file(filepath) -> Self:
            prec = set()
            with open(filepath) as f:
                _njobs, _nress = map(lambda s: int(s), f.readline().split())

                f.readline()

                resubs = list(map(lambda s: int(s), f.readline().split()))

                ress = [Resource(ub=resub) for resub in resubs]

                jobs = []
                job_id = 0
                while (line := f.readline()):
                    datas = list(map(lambda s: int(s), line.split()))
                    if len(datas) == 0:
                        continue

                    idx = 0
                    time = datas[idx]
                    idx += 1
                    res_usages = [datas[idx + jdx] for jdx in range(_nress)]
                    idx += _nress

                    # 次から始まる数値列の長さなのでスキップ
                    idx += 1

                    while idx < len(datas):
                        prec.add((job_id, datas[idx] - 1))
                        idx += 1

                    jobs.append(Job(id=job_id, time=time, res_usages=res_usages))
                    job_id += 1

            return Condition(jobs=jobs, ress=ress, prec=prec)
    return Condition, Job, Resource


@app.cell
def _(Condition, Path, data_dir):
    _filepath = data_dir / Path("pat1.rcp")
    cond1 = Condition.from_file(_filepath)
    return (cond1,)


@app.cell
def _(cond1):
    cond1.model_dump()
    return


@app.cell
def _(Condition, highspy):
    class Model2Highs:
        def __init__(self, cond: Condition):
            self.model = highspy.Highs()

            horizon = sum(job.time for job in cond.jobs)

            # s[j]: ジョブ j の開始期
            self.s = [self.model.addVariable(ub=horizon - 1) for job in cond.jobs]
            # e[j]: ジョブ j の終了期 "の次の期"
            self.e = [self.s[jdx] + job.time for jdx, job in enumerate(cond.jobs)]
            # x[j][t]: ジョブ j を期 t に開始する時だけ 1
            self.x = [[self.model.addBinary() for t in range(horizon)] for job in cond.jobs]

            # 各ジョブは必ず処理される
            for jdx, job in enumerate(cond.jobs):
                self.model.addConstr(
                    sum(self.x[jdx][t] for t in range(horizon - max(0, job.time - 1))) == 1
                )
                for t in range(horizon - max(0, job.time - 1), horizon):
                    self.model.addConstr(self.x[jdx][t] == 0)

            # s と x の関係
            self.model.addConstrs(
                [
                    sum(t * self.x[jdx][t] for t in range(horizon)) == self.s[jdx]
                    for jdx, _ in enumerate(cond.jobs)
                ]
            )

            # ジョブ依存関係
            for idx, jdx in cond.prec:
                self.model.addConstr(self.e[idx] <= self.s[jdx])

            # 資源制約
            for id_res, res in enumerate(cond.ress):
                for t in range(horizon):
                    self.model.addConstr(
                        sum(
                            sum(
                                self.x[id_job][_t]
                                for _t in range(max(0, t - job.time + 1), t + 1)
                            ) * job.res_usages[id_res]
                            for id_job, job in enumerate(cond.jobs)
                        ) <= res.ub
                    )

            # 目的関数: makespan
            self.objective = self.model.addVariable(ub=horizon)
            for jdx, job in enumerate(cond.jobs):
                self.model.addConstr(self.objective >= self.e[jdx])

        def solve(self) -> None:
            # self.model.run()
            self.model.minimize(self.objective)
            self.solution = self.model.getSolution()
            self.info = self.model.getInfo()
    return (Model2Highs,)


@app.cell
def _(Model2Highs, cond1):
    model3 = Model2Highs(cond1)
    model3.solve()
    return (model3,)


@app.cell
def _(model3):
    print (f"Opt.value = {model3.info.objective_function_value}")

    for _id_job, s in enumerate(model3.s):
        _val = model3.solution.col_value[s.index]
        print(f"s[{_id_job}] = {_val}")
    return (s,)


@app.cell
def _(Condition, cp_model):
    class Model2CpSat:
        def __init__(self, cond: Condition):
            self.model = cp_model.CpModel()

            horizon = sum(job.time for job in cond.jobs)

            self.starts = [self.model.new_int_var(lb=0, ub=horizon-job.time, name="") for job in cond.jobs]
            self.jobs = [
                self.model.new_fixed_size_interval_var(self.starts[id_job], job.time, name="")
                for id_job, job in enumerate(cond.jobs)
            ]

            # ジョブ間依存関係
            for idx, jdx in cond.prec:
                self.model.add(self.jobs[idx].end_expr() <= self.jobs[jdx].start_expr())

            # 資源制約
            for id_res, res in enumerate(cond.ress):
                capacity = res.ub
                intervals = []
                demands = []
                for id_job, job in enumerate(cond.jobs):
                    if job.res_usages[id_res] == 0:
                        continue
                    intervals.append(self.jobs[id_job])
                    demands.append(job.res_usages[id_res])

                self.model.add_cumulative(intervals, demands, capacity)

            # 目的関数: makespan
            self.objective = self.model.new_int_var(lb=0, ub=horizon, name="")
            self.model.add_max_equality(self.objective, [interval.end_expr() for interval in self.jobs])
            self.model.minimize(self.objective)

        def solve(self, timeout: int = 180):
            self.solver = cp_model.CpSolver()
            self.solver.parameters.log_search_progress = True
            self.solver.parameters.max_time_in_seconds = timeout
            self.status = self.solver.solve(self.model)
    return (Model2CpSat,)


@app.cell
def _(Model2CpSat, cond1):
    model4 = Model2CpSat(cond1)
    model4.solve()
    return (model4,)


@app.cell
def _(model4):
    print (f"Opt.value = {model4.solver.value(model4.objective)}")

    for _id_job, interval in enumerate(model4.jobs):
        _val = model4.solver.value(interval.start_expr())
        print(f"s[{_id_job}] = {_val}")
    return (interval,)


@app.cell
def _(Condition, Path, data_dir):
    _filepath = data_dir / Path("pat104.rcp")
    cond2 = Condition.from_file(_filepath)
    return (cond2,)


@app.cell
def _(Model2Highs, cond2):
    model5 = Model2Highs(cond2)
    model5.solve()
    return (model5,)


@app.cell
def _(Model2CpSat, cond2):
    model6 = Model2CpSat(cond2)
    model6.solve()
    return (model6,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
