# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "didppy==0.9.0",
#     "marimo",
#     "ortools==9.12.4544",
#     "pydantic==2.11.3",
#     "ruff==0.11.5",
# ]
# ///

import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import random
    from typing import Self
    import pydantic
    from ortools.sat.python import cp_model
    import didppy
    return Self, cp_model, didppy, pydantic, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# タレントスケジューリング問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        映画の各シーンを撮影する. 
        各シーンには出演する役者が決まっており, 
        同じ役者が出演するシーンは同時に撮影することはできない. 
        各役者には初回撮影日から最終撮影日までの日数分のギャラを支払わねばならない. 
        間に撮影の無い日があってもその日の報酬も支払われることになる. 
        この問題では支払うギャラを最小化する.

        - $S = \{ 1, \dots, n \}$: 撮影シーン
        - $A = \{ 1, \dots, m \}$: 役者
        - $A_s \subset A \space (\forall s \in S)$: シーン $s$ を撮るのに必要な役者
        - $d_s \in \mathbb{N} \space (\forall s \in S)$: シーン $s$ を撮るのに必要な日数
        - $c_a \in \mathbb{N} \space (\forall a \in A)$: 役者 $a$ を 1 日拘束することで発生するギャラ
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 実装""")
    return


@app.cell
def _(Self, pydantic, random):
    class Actor(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(frozen=True)
        id: int
        cost: int

    class Scene(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(frozen=True)
        id: int
        time: int
        actors: set[int]

    class Condition(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(frozen=True)
        actors: list[Actor]
        scenes: list[Scene]

        @staticmethod
        def example(n_scene: int, n_actor: int) -> Self:
            random.seed(0)

            actors = [
                Actor(
                    id=i,
                    cost=random.randint(1, 5)
                )
                for i in range(n_actor)
            ]
            scenes = [
                Scene(
                    id=j,
                    time=random.randint(1, 5),
                    actors=random.sample(list(range(n_actor)), random.randint(2, n_actor))
                )
                for j in range(n_scene)
            ]

            return Condition(actors=actors, scenes=scenes)
    return Actor, Condition, Scene


@app.cell
def _(Actor, Condition, Scene):
    # Toy problem from https://github.com/domain-independent-dp/didp-rs/blob/main/didppy/examples/talent-scheduling.ipynb

    _n = 4
    _m = 4

    _actors = [
        Actor(id=0, cost=1),
        Actor(id=1, cost=3),
        Actor(id=2, cost=1),
        Actor(id=3, cost=2),
    ]

    _scenes = [
        Scene(id=0, time=1, actors={0, 1, 3}),
        Scene(id=1, time=1, actors={1, 2}),
        Scene(id=2, time=1, actors={0, 2, 3}),
        Scene(id=3, time=1, actors={0, 1, 2}),
    ]

    cond1 = Condition(actors=_actors, scenes=_scenes)
    return (cond1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Google OR-Tools でのモデリング""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### 決定変数

        - $\text{interval}_s = [\text{start}_s, \text{end}_s) \space (\forall s \in S)$: 各シーンの撮影期間を表す区間変数.
        - $\text{keep}_a \space (\forall a \in A)$: 役者 $a$ が拘束された日数

        #### 目的関数

        \[
            \sum_{a \in A} c_a \text{keep}_a
        \]

        #### 制約

        - $\text{start}_s + d_s = \text{end}_s \space (s \in S)$
        - 役者 $a \in A$ が拘束された日数: $\text{keep}_a = \max \{ \text{end}_s \mid s \in S, \space a \in A_s \} - \min \{ \text{start}_s \mid s \in S, \space a \in A_s \}$
        - 各役者は同時に 1 つのシーンの撮影しかできない:
          $\text{no-overlap} \{ I_s \mid s \in S, \space a \in A_s \}$
            - × $\to$ どうやら撮影は平行に行えないらしい. 課すべき制約は $\text{no-overlap} \{ I_s \mid s \in S\}$
        """
    )
    return


@app.cell
def _(Condition, cp_model):
    class ModelCpSat:
        def __init__(self, cond: Condition):
            self.model = cp_model.CpModel()

            horizon = sum(scene.time for scene in cond.scenes)

            self.starts = [self.model.new_int_var(0, horizon, "") for _ in cond.scenes]
            self.intervals = [
                self.model.new_fixed_size_interval_var(self.starts[id_scene], scene.time, "") 
                for id_scene, scene in enumerate(cond.scenes)
            ]

            # for id_act, actor in enumerate(cond.actors):
            #     self.model.add_no_overlap(
            #         [
            #             self.intervals[id_scene]
            #             for id_scene, scene in enumerate(cond.scenes)
            #             if id_act in scene.actors
            #         ]
            #     )
            self.model.add_no_overlap(self.intervals)

            self.objective = 0
            for id_act, actor in enumerate(cond.actors):
                act_start = self.model.new_int_var(0, horizon, "")
                act_end = self.model.new_int_var(0, horizon, "")
                self.model.add_min_equality(
                    act_start,
                    [
                        interval.start_expr()
                        for scene, interval in zip(cond.scenes, self.intervals)
                        if id_act in scene.actors
                    ]
                )
                self.model.add_max_equality(
                    act_end,
                    [
                        interval.end_expr()
                        for scene, interval in zip(cond.scenes, self.intervals)
                        if id_act in scene.actors
                    ]
                )

                self.objective += actor.cost * (act_end - act_start)

            self.model.minimize(self.objective)

        def solve(self, timeout: int = 180) -> None:
            self.solver = cp_model.CpSolver()
            self.solver.parameters.log_search_progress = True
            self.solver.parameters.max_time_in_seconds = timeout
            self.status = self.solver.solve(self.model)

        def print_solution(self) -> None:
            shoots = [(f"Shoot {i}", self.solver.value(interval.start_expr())) for i, interval in enumerate(self.intervals)]
            shoots.sort(key=lambda x: x[1])

            for name, start in shoots:
                print(f"{name} starts at {start}")
    return (ModelCpSat,)


@app.cell
def _(ModelCpSat, cond1, mo):
    model1 = ModelCpSat(cond1)
    model1.solve()

    mo.md(f"Time: {model1.solver.wall_time:.9f}")
    return (model1,)


@app.cell
def _(model1):
    model1.print_solution()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### didp でのモデリング""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### DP Formulation

        シーンの撮影が途中まで完了し, 残りが $Q \subset S$ であるとする. 
        次に $s \in Q$ を撮る場合, 拘束される役者の集合 $L(s, Q)$ は

        \[
            L(s, Q) = A_s \cup \left( \bigcup_{s' \in S \setminus Q} A_{s'} \cap \bigcup_{s' \in Q} A_{s'} \right)
        \]

        となる. 
        右辺の大括弧内は今までのシーン $S \setminus Q$ の撮影で既に呼び寄せていて, 
        この先も撮影があるため解放できない役者の集合を表す. 
        シーン $s$ の撮影が終わると支払われるギャラが $d_s \sum_{a \in L(s, Q)} c_a$ だけ増加し, 
        $Q$ が $Q \setminus \{ s \}$ で更新される. 

        \begin{align*}
            &\text{compute} & &V(S) \\
            &\text{s.t.} & &V(Q) = \begin{dcases}
                    \min_{s \in Q} \left( d_s \sum_{a \in L(s, Q)} c_a + V(Q \setminus \{s\}) \right) & \text{if } Q \neq \emptyset \\
                    0 & \text{if } Q = \emptyset
                \end{dcases}
        \end{align*}

        $V(Q)$ は $Q$ だけを撮影するコストを表していることに注意.

        #### Force Transition

        シーン $s \in Q$ に必要な役者がすでに現場に拘束されておりかつ, 
        $s$ がその全員を必要とする撮影の場合, つまり

        \[
            A_s = \bigcup_{s' \in S \setminus Q} A_{s'} \cap \bigcup_{s' \in Q} A_{s'} \quad (s \in Q)
        \]

        のとき, シーン $s$ を直ちに撮影するのが最適となる. 
        この場合, 上記の更新規則より高い優先順位で下記の更新を行う. 

        \[
            V(Q) = d_s \sum_{a \in A_s} c_a + V(Q \setminus \{ s \})
        \]

        #### Dual Bound

        ドメイン知識で計算の高速化ができるっぽい. 
        今回のケースだと下記のような制約を入れる. 

        \[
            V(Q) \geq \sum_{s \in Q} d_s \sum_{a \in A_s} c_a
        \]
        """
    )
    return


@app.cell
def _(Condition, didppy):
    class ModelDidp:
        def __init__(self, cond: Condition):
            self.model: didppy.Model = didppy.Model()

            scene_indices: list[int] = list(range(len(cond.scenes)))

            # S, A
            objtype_scene: didppy.ObjectType = self.model.add_object_type(number=len(cond.scenes))
            objtype_actor: didppy.ObjectType = self.model.add_object_type(number=len(cond.actors))

            # Q
            remaining: didppy.SetVar = self.model.add_set_var(
                object_type=objtype_scene,
                target=scene_indices,
            )

            # 各シーンの撮影に必要な役者の集合のリスト
            # didp の example では object_type が scene だったが多分 actor が正しい. 
            scene_to_actors_table: didppy.SetTable1D = self.model.add_set_table(
                [scene.actors for scene in cond.scenes], object_type=objtype_actor
            )

            # 各役者の日ごとのギャラ
            actor_to_cost: didppy.IntTable1D = self.model.add_int_table([actor.cost for actor in cond.actors])

            # 既に来てもらった役者の集合. 現地に留まっているとは限らず, 帰ってる可能性もある. 
            # 必須役者集合のリスト scene_to_actors_table に remaining の補集合を入れることで求める
            came_to_location: didppy.SetExpr = scene_to_actors_table.union(remaining.complement())

            # これから撮影のある役者
            # 必須役者集合のリスト scene_to_actors_table に remaining を入れることで求める
            standby: didppy.SetExpr = scene_to_actors_table.union(remaining)

            # 既に来てもらったことのある役者とこれから撮影のある役者の共通部分. 
            # 次の撮影時にギャラを支払う必要がある. 
            # Define a state function to avoid redundant evaluation of an expensive expression
            on_location: didppy.SetExpr = self.model.add_set_state_fun(came_to_location & standby)

            # Base Case
            self.model.add_base_case(conditions=[remaining.is_empty()], cost=0)

            # Transition
            for s in scene_indices:
                # 残ってもらっている役者と s の撮影に必要な役者の和集合. 
                # 彼らに対してのみギャラを支払う. 
                on_location_s: didppy.SetExpr = scene_to_actors_table[s] | on_location

                shoot: didppy.Transition = didppy.Transition(
                    name=f"shoot {s}",
                    cost=cond.scenes[s].time * actor_to_cost[on_location_s] + didppy.IntExpr.state_cost(),
                    effects=[(remaining, remaining.remove(s))],
                    preconditions=[remaining.contains(s)],
                )
                self.model.add_transition(shoot)

            # Dual Bound: 各シーンを撮影する最低コスト
            scene_to_min_cost: didppy.IntTable1D = self.model.add_int_table(
                [
                    cond.scenes[s].time * sum(
                        cond.actors[a].cost 
                        for a in cond.scenes[s].actors
                    ) for s in scene_indices
                ]
            )
            self.model.add_dual_bound(scene_to_min_cost[remaining])

            # Force Transition
            for s in scene_indices:
                shoot: didppy.Transition = didppy.Transition(
                    name=f"forced shoot {s}",
                    cost=scene_to_min_cost[s] + didppy.IntExpr.state_cost(),
                    effects=[(remaining, remaining.remove(s))],
                    preconditions=[
                        remaining.contains(s),
                        scene_to_actors_table[s] == on_location,
                    ],
                )
                self.model.add_transition(shoot, forced=True)

        def solve(self, timeout=180, threads: int=8) -> None:
            self.solver: didppy.CABS = didppy.CABS(self.model, threads=threads, quiet=False, time_limit=timeout)
            self.solution: didppy.Solution = self.solver.search()

        def print_solution(self) -> None:
            print("Transitions to apply:")
            print("")

            for _t in self.solution.transitions:
                print(_t.name)

            print()
            print(f"Cost: {self.solution.cost}")
    return (ModelDidp,)


@app.cell
def _(ModelDidp, cond1, mo):
    model2 = ModelDidp(cond1)
    model2.solve()
    mo.md(f"Time: {model2.solution.time:.9f}")
    return (model2,)


@app.cell
def _(model2):
    model2.print_solution()
    return


@app.cell
def _(Condition):
    cond2 = Condition.example(n_scene=20, n_actor=10)

    cond2.model_dump()
    return (cond2,)


@app.cell
def _(ModelCpSat, cond2, mo):
    model3 = ModelCpSat(cond2)
    model3.solve()
    mo.md(f"Time: {model3.solver.wall_time:.9f}")
    return (model3,)


@app.cell
def _(model3):
    model3.print_solution()
    return


@app.cell
def _(ModelDidp, cond2, mo):
    model4 = ModelDidp(cond2)
    model4.solve(threads=10)
    mo.md(f"Time: {model4.solution.time:.9f}")
    return (model4,)


@app.cell
def _(model4):
    model4.print_solution()
    return


if __name__ == "__main__":
    app.run()
