# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.3.0",
#     "highspy==1.15.1",
#     "marimo",
#     "pandas==3.0.6",
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
    from pprint import pprint
    import datetime
    import pandas
    import altair
    import highspy

    return Path, altair, datetime, highspy, os, pandas, pprint


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ジョブショップスケジューリング問題 (HiGHS)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    問題設定や他のソルバーでの求解は `jobshop.py` を参照.
    """)
    return


@app.cell
def _(Path, os):
    parent = str(Path(os.path.abspath(__file__)).parent)
    data_dir = os.path.join(parent, "data")
    return (data_dir,)


@app.class_definition
class Task(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    machine: int = pydantic.Field(..., ge=0, frozen=True)
    time: int = pydantic.Field(..., ge=0, frozen=True)


@app.class_definition
class Job(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    tasks: list[Task] = pydantic.Field(frozen=True)

    @classmethod
    def from_file(cls, fname: str) -> list[Self]:
        with open(fname) as f:
            n, m = None, None
            machine, proc_time = {}, {}

            i = 0
            for line in f:
                if line[0] == "#":
                    continue

                if n is None or m is None:
                    n, m = map(int, line.split())
                    print(f"{n=}, {m=}")
                    continue

                L = list(map(int, line.split()))
                for j in range(m):
                    machine[i, j] = L[2 * j]
                    proc_time[i, j] = L[2 * j + 1]
                i += 1

        jobs = []
        for i in range(n):
            tasks = []
            for j in range(m):
                tasks.append(Task(machine=machine[i, j], time=proc_time[i, j]))
            jobs.append(cls(tasks=tasks))

        return jobs


@app.cell
def _(altair, pandas):
    def plot_altair(df: pandas.DataFrame):
        return (
            altair.Chart(df)
            .mark_bar()
            .encode(
                x="start",
                x2="end",
                y="resource",
                color="job",
            )
            .properties(width="container", height=400)
        )

    return (plot_altair,)


@app.cell
def _(data_dir, os, pprint):
    fname1 = os.path.join(data_dir, "ft06.txt")
    jobs1 = Job.from_file(fname1)
    pprint(jobs1)
    return (jobs1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 数理最適化ソルバーによる求解
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    各ジョブに含まれるオペレーション数は機械の数 $m$ に一致すると仮定する.

    \begin{align*}
    &\min &z \\
    &\text{s.t. } & s_{ij} + p_{ij} - M (1 - x_{ijkl}) &\leq s_{kl} \quad &(\forall j \ne k) \\
    & & x_{ijkl} + x_{klij} &= 1 \quad &((i,j) \ne (k,l) \land \text{machine} (i,j) = \text{machine} (k,l)) \\
    & & s_{ij} + p_{ij} &\le s_{i,j+1} \quad &(\forall i, j = 1, \dots, m-1) \\
    & & s_{im} &\le z \quad &(\forall i) \\
    & & s_{i1} &\ge 0 \quad &(\forall i) \\
    & & x_{ijkl} &\in \{ 0, 1 \} \quad &(\forall (i,j) \ne (k,l))
    \end{align*}
    """)
    return


@app.cell
def _(datetime, highspy, pandas):
    class _MyInterval:
        def __init__(self, model: highspy.Highs, lb: int, ub: int, proctime: int):
            self.lb = lb
            self.ub = ub
            self.start = model.addVariable(lb=lb, ub=ub - proctime)
            self.time = proctime
            self.end = self.start + self.time


    def _my_add_no_overlap(model: highspy.Highs, tasks: list[_MyInterval]) -> None:
        for idx1, task1 in enumerate(tasks):
            for idx2, task2 in enumerate(tasks):
                if idx1 >= idx2:
                    continue

                big_m = max(task1.ub - task2.lb, task2.ub - task1.lb)
                tmp1 = model.addBinary()  # [ task1 ] [ task2 ] の順
                tmp2 = model.addBinary()  # [ task2 ] [ task1 ] の順
                model.addConstrs(
                    [
                        task1.end - big_m * (1 - tmp1) <= task2.start,
                        task2.end - big_m * (1 - tmp2) <= task1.start,
                        tmp1 + tmp2 == 1,
                    ]
                )


    class ModelHighs:
        def __init__(self, jobs: list[Job]):
            self.jobs = jobs
            num_machines = len(
                set(task.machine for job in self.jobs for task in job.tasks)
            )
            self.machines = list(range(num_machines))

            self.model = highspy.Highs()

            self.intervals = [[None for task in job.tasks] for job in jobs]
            machine_to_interval = {m: [] for m in self.machines}

            horizon = sum(task.time for job in self.jobs for task in job.tasks)
            for id_job, job in enumerate(self.jobs):
                for id_task, task in enumerate(job.tasks):
                    interval = _MyInterval(self.model, 0, horizon, task.time)
                    self.intervals[id_job][id_task] = interval
                    machine_to_interval[task.machine].append(interval)

            for machine in machine_to_interval:
                if len(machine_to_interval[machine]) > 0:
                    _my_add_no_overlap(self.model, machine_to_interval[machine])

            for id_job, job in enumerate(self.jobs):
                for id_task, task in enumerate(job.tasks):
                    if id_task > 0:
                        curr = self.intervals[id_job][id_task]
                        prev = self.intervals[id_job][id_task - 1]
                        self.model.addConstr(curr.start >= prev.end)

            makespan = self.model.addVariable(lb=0, ub=horizon)
            self.model.addConstrs(
                [
                    self.intervals[id_job][-1].end <= makespan
                    for id_job, job in enumerate(self.jobs)
                ],
            )
            self.model.minimize(makespan)

        def solve(self) -> None:
            self.model.run()
            self.solution = self.model.getSolution()

        def to_df(self) -> pandas.DataFrame:
            today = datetime.date.today()
            l = []
            for id_job, job in enumerate(self.jobs):
                for id_task, task in enumerate(job.tasks):
                    start = self.solution.col_value[
                        self.intervals[id_job][id_task].start.index
                    ]
                    start = round(start)
                    end = start + self.jobs[id_job].tasks[id_task].time
                    l.append(
                        dict(
                            job=f"job{id_job}",
                            task=f"task{id_task}",
                            resource=f"machine{self.jobs[id_job].tasks[id_task].machine}",
                            start=today + datetime.timedelta(start),
                            end=today + datetime.timedelta(end),
                        )
                    )
            df = pandas.DataFrame(l)
            df["start"] = pandas.to_datetime(df["start"])
            df["end"] = pandas.to_datetime(df["end"])
            return df

    return (ModelHighs,)


@app.cell
def _(ModelHighs, jobs1):
    model1_highs = ModelHighs(jobs1)
    model1_highs.solve()
    return (model1_highs,)


@app.cell
def _(model1_highs, plot_altair):
    plot_altair(model1_highs.to_df())
    return


if __name__ == "__main__":
    app.run()
