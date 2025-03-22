# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "ortools==9.12.4544",
#     "pandas==2.2.3",
#     "plotly==6.0.1",
#     "pyarrow==19.0.1",
#     "pydantic==2.10.6",
# ]
# ///

import marimo

__generated_with = "0.11.25"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    from pathlib import Path
    from pprint import pprint
    from typing import Self
    import datetime
    import pandas, pyarrow
    import pydantic
    import plotly.express
    import altair
    from ortools.sat.python import cp_model
    return (
        Path,
        Self,
        altair,
        cp_model,
        datetime,
        os,
        pandas,
        plotly,
        pprint,
        pyarrow,
        pydantic,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# ジョブショップスケジューリング問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $J || C_{\max}$ と書く. 

        - ジョブ $J_1, \dots, J_n$
        - ジョブ $J_j$ に属するオペレーション $O_{1j}, \dots, O_{m_jj}$. この順で処理される. 
        - 機械 $M_1, \dots, M_m$
        - オペレーション $O_{ij}$ は機械 $\mu_{ij}$ で作業時間 $p_{ij}$ かけて処理する. 
        - オペレーションは中断できない
        - 最後のオペレーションの終了時刻を最小化
        """
    )
    return


@app.cell
def _(Path, __file__, os):
    parent = str(Path(os.path.abspath(__file__)).parent)
    data_dir = os.path.join(parent, "data")
    return data_dir, parent


@app.cell
def _(pydantic):
    class Task(pydantic.BaseModel):
        machine: int = pydantic.Field(..., ge=0, frozen=True)
        time: int = pydantic.Field(..., ge=0, frozen=True)
    return (Task,)


@app.cell
def _(Self, Task, pydantic):
    class Job(pydantic.BaseModel):
        tasks: list[Task] = pydantic.Field(frozen=True)

        def from_file(fname: str) -> list[Self]:
            with open(fname) as f:
                lines = f.readlines()

            n, m = map(int, lines[0].split())
            print(f"{n=}, {m=}")

            machine, proc_time = {}, {}
            for i in range(n):
                L = list(map(int, lines[i + 1].split()))
                for j in range(m):
                    machine[i, j] = L[2 * j]
                    proc_time[i, j] = L[2 * j + 1]

            jobs = []
            for i in range(n):
                tasks = []
                for j in range(m):
                    tasks.append(Task(machine=machine[i, j], time=proc_time[i, j]))
                jobs.append(Job(tasks=tasks))

            return jobs
    return (Job,)


@app.cell
def _(Job, cp_model, datetime, pandas):
    class Model:
        def __init__(self, jobs: list[Job]):
            self.jobs = jobs
            self.model = cp_model.CpModel()
            num_machines = len(set(task.machine for job in self.jobs for task in job.tasks))
            self.machines = list(range(num_machines))
            horizon = sum(task.time for job in self.jobs for task in job.tasks)

            self.starts = [[None for task in job.tasks] for job in jobs]
            self.intervals = [[None for task in job.tasks] for job in jobs]
            machine_to_interval = {m: [] for m in self.machines}

            for id_job, job in enumerate(self.jobs):
                for id_task, task in enumerate(job.tasks):
                    suffix = f"_{id_job}_{id_task}"
                    start = self.model.new_int_var(0, horizon, "start" + suffix)
                    interval = self.model.new_fixed_size_interval_var(start, task.time, "interval" + suffix)
                    self.starts[id_job][id_task] = start
                    self.intervals[id_job][id_task] = interval
                    machine_to_interval[task.machine].append(interval)

            for machine in machine_to_interval:
                if len(machine_to_interval[machine]) > 0:
                    self.model.add_no_overlap(machine_to_interval[machine])

            for id_job, job in enumerate(self.jobs):
                for id_task, task in enumerate(job.tasks):
                    if id_task > 0:
                        curr = self.intervals[id_job][id_task]
                        prev = self.intervals[id_job][id_task - 1]
                        self.model.add(curr.start_expr() >= prev.end_expr())

            makespan = self.model.new_int_var(0, horizon, "makespan")
            self.model.add_max_equality(
                makespan,
                [self.intervals[id_job][-1].end_expr() for id_job, job in enumerate(self.jobs)],
            )
            self.model.minimize(makespan)

        def solve(self):
            self.solver = cp_model.CpSolver()
            self.solver.parameters.log_search_progress = True
            self.status = self.solver.solve(self.model)

        def to_df(self) -> pandas.DataFrame:
            today = datetime.date.today()
            l = []
            for id_job, job in enumerate(self.jobs):
                for id_task, task in enumerate(job.tasks):
                    start = self.solver.value(self.intervals[id_job][id_task].start_expr())
                    end = start + self.jobs[id_job].tasks[id_task].time
                    l.append(
                        dict(
                            job=f"job{id_job}",
                            task=f"task{id_task}",
                            resource=f"machine{self.jobs[id_job].tasks[id_task].machine}",
                            start=today + datetime.timedelta(start),
                            end=today + datetime.timedelta(end)
                        )
                    )
            df = pandas.DataFrame(l)
            df["start"] = pandas.to_datetime(df["start"])
            df["end"] = pandas.to_datetime(df["end"])
            return df
    return (Model,)


@app.cell
def _(pandas, plotly):
    def plot_plotly(df: pandas.DataFrame):
        return plotly.express.timeline(
            df,
            x_start="start",
            x_end="end",
            y="resource",
            color="job",
            opacity=0.5
        )
    return (plot_plotly,)


@app.cell
def _(altair, pandas):
    def plot_altair(df: pandas.DataFrame):
        return altair.Chart(df).mark_bar().encode(
            x="start",
            x2="end",
            y="resource",
            color="job",
        ).properties(
            width="container",
            height=400
        )
    return (plot_altair,)


@app.cell
def _(Job, data_dir, os, pprint):
    fname1 = os.path.join(data_dir, "ft06.txt")
    jobs1 = Job.from_file(fname1)
    pprint(jobs1)
    return fname1, jobs1


@app.cell
def _(Model, jobs1):
    model1_cpsat = Model(jobs1)
    model1_cpsat.solve()
    return (model1_cpsat,)


@app.cell
def _(model1_cpsat, plot_plotly):
    plot_plotly(model1_cpsat.to_df())
    return


@app.cell
def _(mo, model1_cpsat, plot_altair):
    mo.ui.altair_chart(plot_altair(model1_cpsat.to_df()))
    return


@app.cell
def _(model1_cpsat, plot_altair):
    plot_altair(model1_cpsat.to_df())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
