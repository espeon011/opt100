# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "ortools==9.15.6755",
#     "pydantic==2.13.5",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from typing import Self

    import pydantic


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    from pathlib import Path
    from ortools.sat.python import cp_model

    return Path, cp_model, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 資源制約付きプロジェクトスケジューリング問題 (CP-SAT)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    定式化や HiGHS での求解は `resource_constrained.py` を参照.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## インスタンス

    kobe-scheduling の patterson.rcp を使う. フォーマットは `resource_constrained.py` に書いた通り (多分).
    """)
    return


@app.cell
def _(Path, os):
    parent = str(Path(os.path.abspath(__file__)).parent)
    data_dir = Path(parent, "kobe-scheduling", "data", "rcpsp", "patterson.rcp")
    return (data_dir,)


@app.class_definition
class Job(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
    id: int
    time: int
    res_usages: list[int]


@app.class_definition
class Resource(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
    ub: int


@app.class_definition
class Condition(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
    jobs: list[Job]
    ress: list[Resource]
    prec: set[tuple[int, int]]

    @classmethod
    def from_file(cls, filepath) -> Self:
        prec = set()
        with open(filepath) as f:
            _njobs, _nress = map(lambda s: int(s), f.readline().split())

            f.readline()

            resubs = list(map(lambda s: int(s), f.readline().split()))

            ress = [Resource(ub=resub) for resub in resubs]

            jobs = []
            job_id = 0
            while line := f.readline():
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

        return cls(jobs=jobs, ress=ress, prec=prec)


@app.cell
def _(Path, data_dir):
    _filepath = data_dir / Path("pat1.rcp")
    cond1 = Condition.from_file(_filepath)
    return (cond1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CP-SAT によるモデリング
    """)
    return


@app.cell
def _(cp_model):
    class Model2CpSat:
        def __init__(self, cond: Condition):
            self.model = cp_model.CpModel()

            horizon = sum(job.time for job in cond.jobs)

            self.starts = [
                self.model.new_int_var(lb=0, ub=horizon - job.time, name="")
                for job in cond.jobs
            ]
            self.jobs = [
                self.model.new_fixed_size_interval_var(
                    self.starts[id_job], job.time, name=""
                )
                for id_job, job in enumerate(cond.jobs)
            ]

            # ジョブ間依存関係
            for idx, jdx in cond.prec:
                self.model.add(
                    self.jobs[idx].end_expr() <= self.jobs[jdx].start_expr()
                )

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
            self.model.add_max_equality(
                self.objective, [interval.end_expr() for interval in self.jobs]
            )
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
    print(f"Opt.value = {model4.solver.value(model4.objective)}")

    for _id_job, interval in enumerate(model4.jobs):
        _val = model4.solver.value(interval.start_expr())
        print(f"s[{_id_job}] = {_val}")
    return


@app.cell
def _(Path, data_dir):
    _filepath = data_dir / Path("pat104.rcp")
    cond2 = Condition.from_file(_filepath)
    return (cond2,)


@app.cell
def _(Model2CpSat, cond2):
    model6 = Model2CpSat(cond2)
    model6.solve()
    return


if __name__ == "__main__":
    app.run()
