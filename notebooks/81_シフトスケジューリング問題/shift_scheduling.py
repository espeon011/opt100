# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "nbformat==5.10.4",
#     "networkx==3.5",
#     "ortools==9.13.4784",
#     "pydantic==2.11.7",
#     "ruff==0.11.9",
# ]
# ///

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    import pathlib
    from typing import Self
    import pydantic
    import networkx as nx
    from ortools.sat.python import cp_model
    return Self, cp_model, nx, os, pathlib, pydantic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# シフト最適化問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 問題設定""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    各ジョブは予め実行時間が与えられている. 
    その上で作業員を各ジョブに割り当て, 同じ作業員が同一時刻に複数のタスクを処理しないようにする.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 定数

    - $J$: ジョブの集合
    - $W$: 作業員の集合
    - $b_w$: 作業員 $w$ の固定費用
    - $s_j, f_j$: ジョブ $j$ の開始時刻と終了時刻
    - $J_w$: 作業員 $w$ が遂行できるジョブの集合
    - $W_j$: ジョブ $j$ を遂行できる作業員の集合
    - $G = (J, E)$: ジョブを点, ジョブの作業時間に重なりがあるとき点の間に線を張るというルールで作成したグラフ
    - $C$: 区間グラフ $G$ に対する極大クリークの集合
    - $J_c$: クリーク $c$ に含まれるジョブの集合

    ### 決定変数

    - $x_{jw} \in \{ 0, 1 \}$: ジョブ $j$ に作業員 $w$ を割り当てる時 $1$, それ以外のとき $0$. 
    - $y_w \in \{ 0, 1 \}$: 作業員 $w$ が使われるとき $1$, それ以外のとき $0$. 

    ### 制約条件

    - $\sum_{w \in W} x_{jw} = 1 \quad (\forall j \in J)$: 各ジョブは 1 人の作業員によって必ず遂行される. 
    - $\sum_{j \in J_c \cap J_w} x_{jw} \leq y_w \quad (\forall w \in W, \space \forall c \in C)$: 各作業員は同時に 2 つ以上のジョブを処理できない. 

    ### 目的関数

    - $\min \sum_{w \in W} b_w y_w$: 作業員の固定費用の総和を最小化
    """
    )
    return


@app.cell
def _(os, pathlib):
    current_dir = pathlib.Path(os.path.dirname(__file__))
    data_dir = current_dir / "data" / "ptask"
    return (data_dir,)


@app.cell
def _(data_dir):
    fname1 = data_dir / "data_10_51_111_66.dat"

    with open(fname1) as f:
        _lines = f.readlines()

    _lines
    return (fname1,)


@app.cell
def _(Self, pydantic):
    class Job(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(frozen=True)

        start: int
        finish: int

        def intersect_with(self, other: Self) -> bool:
            if self.start <= other.start and self.finish > other.start:
                return True
            if other.start <= self.start and other.finish > self.start:
                return True

            return False
    return (Job,)


@app.cell
def _(pydantic):
    class Worker(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(frozen=True)

        cost: int
        jobs: list[int]
    return (Worker,)


@app.cell
def _(Job, Self, Worker, os, pydantic):
    class Condition(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(frozen=True)

        jobs: list[Job]
        workers: list[Worker]

        @staticmethod
        def from_file(path: str | os.PathLike) -> Self:
            jobs = []
            workers = []

            with open(path) as f:
                lines = f.readlines()
            idx = 0
            num_jobs = None
            while idx < len(lines):
                line = lines[idx]
                if line.startswith("Jobs = "):
                    num_jobs: int = int(line.split()[2])
                    print(f"{num_jobs=}")
                    for _ in range(num_jobs):
                        idx += 1
                        line = lines[idx]
                        start, finish = line.strip().split()
                        jobs.append(Job(start=start, finish=finish))
                if line.startswith("Qualifications = "):
                    num_workers: int = int(line.split()[2])
                    print(f"{num_workers=}")
                    for _ in range(num_workers):
                        idx += 1
                        line = lines[idx]
                        bw, jw = line.strip().split(":")
                        workers.append(Worker(cost=bw, jobs=jw.strip().split()))

                idx += 1

            return Condition(jobs=jobs, workers=workers)
    return (Condition,)


@app.cell
def _(Condition, fname1):
    cond1 = Condition.from_file(fname1)
    return (cond1,)


@app.cell
def _(Condition, cp_model, nx):
    class Model:
        def __init__(self, cond: Condition):
            # グラフの定義
            g = nx.Graph()
            for id_job1, job1 in enumerate(cond.jobs):
                for id_job2, job2 in enumerate(cond.jobs):
                    if id_job2 <= id_job1:
                        continue
                    if job1.intersect_with(job2):
                        g.add_edge(id_job1, id_job2)

            # クリークの列挙
            cliques = [set(c) for c in nx.find_cliques(g)]

            model = cp_model.CpModel()

            x = [[model.new_bool_var("") for _ in cond.workers] for _ in cond.jobs]
            y = [model.new_bool_var("") for _ in cond.workers]

            # x と y の関係
            for idw, worker in enumerate(cond.workers):
                model.add_max_equality(
                    y[idw],
                    [x[idj][idw] for idj, _ in enumerate(cond.jobs)],
                )

            # worker ごとに担当できる job はきまっている
            for idj, job in enumerate(cond.jobs):
                for idw, worker in enumerate(cond.workers):
                    if idj not in worker.jobs:
                        model.add(x[idj][idw] == 0)

            # 各 job は 1 人の worker によって処理される
            for idj, _ in enumerate(cond.jobs):
                model.add_exactly_one(
                    [x[idj][idw] for idw, _ in enumerate(cond.workers)]
                )

            # 同一クリークに属する job は同じ worker で処理できない
            for clique in cliques:
                for idw, worker in enumerate(cond.workers):
                    model.add_at_most_one([x[idj][idw] for idj in clique])

            # model.minimize(
            #     sum(
            #         y[idw]
            #         for idw, _ in enumerate(cond.workers)
            #     )
            # )

            # 解説ページでは worker 数を最小化していたが,
            # worker のコスト込で最小化するのが正しい.
            model.minimize(
                sum(
                    y[idw] * worker.cost for idw, worker in enumerate(cond.workers)
                )
            )

            self.model = model
            self.x = x
            self.y = y
            self.cliques = cliques

        def solve(self, timeout: int = 10):
            self.solver = cp_model.CpSolver()
            self.solver.parameters.log_search_progress = True
            self.solver.parameters.max_time_in_seconds = timeout
            self.status = self.solver.solve(self.model)
    return (Model,)


@app.cell
def _(Model, cond1, mo):
    model1 = Model(cond1)
    model1.solve()

    mo.md(f"Optimal Value = {model1.solver.objective_value}")
    return


@app.cell
def _(Condition, data_dir):
    fname2 = data_dir / "data_40_138_360_33.dat"
    cond2 = Condition.from_file(fname2)
    return (cond2,)


@app.cell
def _(Model, cond2, mo):
    model2 = Model(cond2)
    model2.solve(timeout=60)

    mo.md(f"Optimal Value = {model2.solver.objective_value}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
