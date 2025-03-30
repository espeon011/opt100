# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "amplify-sched==0.2.1",
#     "highspy==1.10.0",
#     "marimo",
#     "ortools==9.12.4544",
#     "pandas==2.2.3",
#     "plotly==6.0.1",
#     "pyarrow==19.0.1",
#     "pydantic==2.11.1",
#     "python-dotenv==1.1.0",
# ]
# ///

import marimo

__generated_with = "0.11.31"
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
    import random
    import dotenv
    import pandas, pyarrow
    import pydantic
    import plotly.express
    import altair
    from ortools.sat.python import cp_model
    import highspy
    import amplify_sched
    return (
        Path,
        Self,
        altair,
        amplify_sched,
        cp_model,
        datetime,
        dotenv,
        highspy,
        os,
        pandas,
        plotly,
        pprint,
        pyarrow,
        pydantic,
        random,
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
        model_config = pydantic.ConfigDict(frozen=True)

        machine: int = pydantic.Field(..., ge=0, frozen=True)
        time: int = pydantic.Field(..., ge=0, frozen=True)
    return (Task,)


@app.cell
def _(Self, Task, pydantic):
    class Job(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(frozen=True)

        tasks: list[Task] = pydantic.Field(frozen=True)

        def from_file(fname: str) -> list[Self]:
            with open(fname) as f:
                n, m = None, None
                machine, proc_time = {}, {}

                i = 0
                for line in f:
                    if line[0] == '#':
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

            #     lines = f.readlines()

            # n, m = map(int, lines[0].split())
            # print(f"{n=}, {m=}")

            # machine, proc_time = {}, {}
            # for i in range(n):
            #     L = list(map(int, lines[i + 1].split()))
            #     for j in range(m):
            #         machine[i, j] = L[2 * j]
            #         proc_time[i, j] = L[2 * j + 1]

            jobs = []
            for i in range(n):
                tasks = []
                for j in range(m):
                    tasks.append(Task(machine=machine[i, j], time=proc_time[i, j]))
                jobs.append(Job(tasks=tasks))

            return jobs
    return (Job,)


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
        ).update_yaxes(categoryorder="category descending")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## OR-Tools による求解""")
    return


@app.cell
def _(Job, cp_model, datetime, pandas):
    class ModelCpSat:
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

        def solve(self, timeout: int = 180):
            self.solver = cp_model.CpSolver()
            self.solver.parameters.log_search_progress = True
            self.solver.parameters.max_time_in_seconds = timeout
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
    return (ModelCpSat,)


@app.cell
def _(ModelCpSat, jobs1):
    model1_cpsat = ModelCpSat(jobs1)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 数理最適化ソルバーによる求解""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(Job, datetime, highspy, pandas):
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
                model.addConstrs([
                    task1.end - big_m * (1 - tmp1) <= task2.start,
                    task2.end - big_m * (1 - tmp2) <= task1.start,
                    tmp1 + tmp2 == 1,
                ])

    class ModelHighs:
        def __init__(self, jobs: list[Job]):
            self.jobs = jobs
            num_machines = len(set(task.machine for job in self.jobs for task in job.tasks))
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
                    start = self.solution.col_value[self.intervals[id_job][id_task].start.index]
                    start = round(start)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## FIXSTARS Amplify Scheduling Engine による求解""")
    return


@app.cell
def _(Job, amplify_sched, datetime, dotenv, os, pandas):
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    token = os.environ["FIXSTARS_SE"]

    class ModelAmplifySe:
        def __init__(self, jobs: list[Job]):
            self.jobs = jobs
            num_machines = len(set(task.machine for job in self.jobs for task in job.tasks))
            self.machines = list(range(num_machines))
            self.se_machines = [amplify_sched.Machine(name=f"machine{midx}") for midx in self.machines]

            self.model = amplify_sched.Model()

            for semachine in self.se_machines:
                self.model.machines.add(machine=semachine)

            self.se_jobs = [amplify_sched.Job(name=f"job{jidx}") for jidx, _ in enumerate(self.jobs)]
            for idx, job in enumerate(self.jobs):
                sejob = self.se_jobs[idx]
                self.model.jobs.add(sejob)
                for jdx, task in enumerate(job.tasks):
                    semachine = self.se_machines[task.machine]
                    setask = amplify_sched.Task()
                    setask.processing_times[semachine] = task.time
                    self.model.jobs[sejob.name].append(setask)

        def solve(self, timeout: int = 1) -> None:
            self.solution = self.model.solve(token=token, timeout=timeout)

        def get_makespan(self) -> int:
            return int(self.solution.table["Finish"].max())

        def to_df(self) -> pandas.DataFrame:
            sol_df = self.solution.table
            today = datetime.date.today()
            l = []
            for id_job, job in enumerate(self.jobs):
                sejob = self.se_jobs[id_job]
                for id_task, task in enumerate(job.tasks):
                    start = int(sol_df[sol_df["Job"] == sejob.name]["Start"].reset_index(drop=True)[id_task])
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
    return ModelAmplifySe, token


@app.cell
def _(ModelAmplifySe, jobs1):
    model1_amplify = ModelAmplifySe(jobs1)
    model1_amplify.solve()

    print(f"makespan = {model1_amplify.get_makespan()}")
    return (model1_amplify,)


@app.cell
def _(model1_amplify):
    model1_amplify.solution.timeline(machine_view=True)
    return


@app.cell
def _(model1_amplify, plot_altair):
    plot_altair(model1_amplify.to_df())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## より大きい問題""")
    return


@app.cell
def _(Job, Task, random):
    def gen_data(n_jobs: int, n_machines: int) -> list[Job]:
        random.seed(0)
        jobs = []
        for id_job in range(n_jobs):
            machines = list(range(n_machines))
            random.shuffle(machines)
            tasks = []
            for id_task in range(n_machines):
                machine = machines[id_task]
                time = random.randint(1, 10)
                tasks.append(Task(machine=machine, time=time))

            jobs.append(Job(tasks=tasks))

        return jobs
    return (gen_data,)


@app.cell
def _(gen_data, pprint):
    jobs2 = gen_data(45, 15)
    pprint(jobs2)
    return (jobs2,)


@app.cell
def _(ModelCpSat, jobs2):
    model2_cpsat = ModelCpSat(jobs2)
    model2_cpsat.solve()
    return (model2_cpsat,)


@app.cell
def _(model2_cpsat, plot_plotly):
    plot_plotly(model2_cpsat.to_df())
    return


@app.cell
def _():
    #model2_highs = ModelHighs(jobs2)
    #model2_highs.solve()
    return


@app.cell
def _():
    #plot_altair(model2_highs.to_df())
    return


@app.cell
def _(ModelAmplifySe, jobs2):
    model2_amplify = ModelAmplifySe(jobs2)
    model2_amplify.solve(timeout=5)

    print(f"makespan = {model2_amplify.get_makespan()}")
    return (model2_amplify,)


@app.cell
def _(model2_amplify, plot_plotly):
    plot_plotly(model2_amplify.to_df())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 他のインスタンス""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ta50 は最適解は知られていない. 

        bounds

        - upper: 1923
        - lower: 1833
        """
    )
    return


@app.cell
def _(os, parent):
    instance_dir = os.path.join(parent, "jsplib/instances")
    return (instance_dir,)


@app.cell
def _(Job, instance_dir, os):
    fname3 = os.path.join(instance_dir, "ta50")
    jobs3 = Job.from_file(fname3)
    return fname3, jobs3


@app.cell
def _(ModelAmplifySe, jobs3, mo):
    model3_amplify = ModelAmplifySe(jobs3)
    model3_amplify.solve(timeout=10)

    mo.md(f"makespan = {model3_amplify.get_makespan()}")
    return (model3_amplify,)


@app.cell
def _(model3_amplify, plot_plotly):
    plot_plotly(model3_amplify.to_df())
    return


@app.cell
def _(ModelCpSat, jobs3, mo):
    model3_cpsat = ModelCpSat(jobs3)
    model3_cpsat.solve(timeout=180)

    mo.md(f"makespan = {round(model3_cpsat.solver.objective_value)}")
    return (model3_cpsat,)


@app.cell
def _(model3_cpsat, plot_plotly):
    plot_plotly(model3_cpsat.to_df())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
