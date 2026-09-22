# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "nbformat==5.10.4",
#     "ortools==9.14.6206",
# ]
# ///

import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from ortools.sat.python import cp_model
    return (cp_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 看護師スケジューリング問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 問題設定""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    典型的な設定だと下記. 

    - 毎日の各勤務 (昼, 夕, 夜) の必要人数
    - 各看護師に対して 30 日間の勤務日数の上下限
    - 指定休日, 指定会議日
    - 連続 7 日間に最低 1 日の休日, 最低 1 日の昼勤務
    - 禁止パターン
        - 3 連続夜勤
        - 4 連続夕勤
        - 5 連続昼勤
        - 夜勤明けの休日以外
        - 夕勤の直後の昼勤あるいは会議
        - 休日 ::lucide:arrow-right:: 勤務 ::lucide:arrow-right:: 休日
    - 夜勤は 2 回連続で行う
    - 2 つのチームの人数をできるだけ均等化
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 定数

    - $W$: 休日以外のシフトの集合
    - $N$: 夜勤以外のシフトの集合
    - $L_{d s}$: 日 $d$ のシフト $s$ の必要人数
    - $\mathrm{LB}, \mathrm{UB}$: 各看護師の 30 日間の勤務日数の上下限
    - $R_i$: 看護師 $i$ が休日を希望する日の集合
    - $T_1, T_2$: チーム 1, チーム 2.

    ### 決定変数

    - $x_{i d s}$: 看護師 $i$ の $d$ 日の勤務が $s$ であるとき $1$. そうでないとき $0$

    ### 制約条件

    - 毎日の各勤務の必要人数: $\sum_{i} x_{i d s} \geq L_{d s}$ $(\forall d, s)$
    - 各看護師の 30 日間の勤務日数の上下限: $\mathrm{LB} \leq \sum_{d, s \in W} x_{i d s} \leq \mathrm{UB}$ $(\forall i)$
    - 指定休日・指定会議日: $\sum_{d \in R_i, s \in W} x_{i d s} \leq 0$ $(\forall i)$
    - 連続 7 日間に最低 1 日の休日, 最低 1 日の昼勤:
        - $\sum_{t = 0}^{6} x_{i, d+t, \text{休}} \geq 1$ $(\forall i, d)$
        - $\sum_{t = 0}^{6} x_{i, d+t, \text{昼}} \geq 1$ $(\forall i, d)$
    - 禁止パターン
        - 3 連続夜勤: $\sum_{t = 0}^{2} x_{i, d+t, \text{夜}} \leq 2$ $(\forall i, d)$
        - 4 連続夕勤: $\sum_{t = 0}^{3} x_{i, d+t, \text{夕}} \leq 3$ $(\forall i, d)$
        - 5 連続昼勤: $\sum_{t = 0}^{4} x_{i, d+t, \text{昼}} \leq 4$ $(\forall i, d)$
        - 夜勤明けの休日以外: $\sum_{s \in W \cap N} x_{i, d + 1, s} <= 5 \cdot (1 - x_{i, d, \text{夜}})$
        - 夕の直後の昼あるいは会議: 会議がよくわからないので省略
        - 休日 ::lucide:arrow-right:: 勤務 ::lucide:arrow-right:: 休日: $x_{i, d-1, \text{休}} + \sum_{s \in W} x_{i d s} + x_{i, d+1, \text{休}} \leq 2$
    - 夜勤は 2 連続: $\sum_{s \in N} x_{i,d-1,s} + x_{i,d,\text{夜}} + \sum_{s \in N} x_{i,d+1,s} \leq 2$
    - 2 チームの人数をできるだけ均等化:
      $\sum_{i \in T_1} x_{i d s} = \sum_{i \in T_2} x_{i d s}$ $(\forall d, s)$. これは制約というよりは目的関数?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## OR-Tools Example""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""参考: https://github.com/google/or-tools/blob/stable/examples/notebook/sat/nurses_sat.ipynb""")
    return


@app.cell
def _(cp_model):
    class NursesPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
        """Print intermediate solutions."""

        def __init__(self, shifts, num_nurses, num_days, num_shifts, limit):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._shifts = shifts
            self._num_nurses = num_nurses
            self._num_days = num_days
            self._num_shifts = num_shifts
            self._solution_count = 0
            self._solution_limit = limit

        def on_solution_callback(self):
            self._solution_count += 1
            print(f"Solution {self._solution_count}")
            for d in range(self._num_days):
                print(f"Day {d}")
                for n in range(self._num_nurses):
                    is_working = False
                    for s in range(self._num_shifts):
                        if self.value(self._shifts[(n, d, s)]):
                            is_working = True
                            print(f"  Nurse {n} works shift {s}")
                    if not is_working:
                        print(f"  Nurse {n} does not work")
            if self._solution_count >= self._solution_limit:
                print(f"Stop search after {self._solution_limit} solutions")
                self.stop_search()

        def solutionCount(self):
            return self._solution_count
    return (NursesPartialSolutionPrinter,)


@app.cell
def _(NursesPartialSolutionPrinter, cp_model):
    class Model:
        def __init__(self, num_nurses: int, num_shifts: int, num_days: int):
            all_nurses = range(num_nurses)
            all_shifts = range(num_shifts)
            all_days = range(num_days)

            model = cp_model.CpModel()

            # shifts[(n, d, s)]: nurse 'n' works shift 's' on day 'd'.
            shifts = {
                (_n, _d, _s): model.new_bool_var(f"shift_n{_n}_d{_d}_s{_s}")
                for _n in all_nurses
                for _d in all_days
                for _s in all_shifts
            }

            # Each shift is assigned to exactly one nurse in the schedule period.
            for _d in all_days:
                for _s in all_shifts:
                    model.add_exactly_one(
                        shifts[(_n, _d, _s)] for _n in all_nurses
                    )

            # Each nurse works at most one shift per day.
            for _n in all_nurses:
                for _d in all_days:
                    model.add_at_most_one(
                        shifts[(_n, _d, _s)] for _s in all_shifts
                    )

            # Try to distribute the shifts evenly, so that each nurse works
            # min_shifts_per_nurse shifts. If this is not possible, because the total
            # number of shifts is not divisible by the number of nurses, some nurses will
            # be assigned one more shift.
            min_shifts_per_nurse = (num_shifts * num_days) // num_nurses
            if num_shifts * num_days % num_nurses == 0:
                max_shifts_per_nurse = min_shifts_per_nurse
            else:
                max_shifts_per_nurse = min_shifts_per_nurse + 1

            for _n in all_nurses:
                shifts_worked = []
                for _d in all_days:
                    for _s in all_shifts:
                        shifts_worked.append(shifts[(_n, _d, _s)])
                model.add(min_shifts_per_nurse <= sum(shifts_worked))
                model.add(sum(shifts_worked) <= max_shifts_per_nurse)

            # 追加
            # shift 0 を夜勤として
            # - 夜勤は 2 連続しなければならない
            # - 夜勤は 3 連続してはいけない
            # ただこれだと夜勤が偏ってしまうので追加の制約が必要
            for _n in all_nurses:
                model.add_implication(shifts[(_n, 0, 0)], shifts[(_n, 1, 0)])
                for _d in all_days:
                    if _d == 0 or _d == num_days - 1:
                        continue
                    model.add_implication(
                        shifts[(_n, _d, 0)], shifts[(_n, _d + 1, 0)]
                    ).only_enforce_if(~shifts[(_n, _d - 1, 0)])
                    model.add_bool_or(
                        [~shifts[(_n, _d + _t, 0)] for _t in [-1, 0, 1]]
                    )

            self.num_nurses = num_nurses
            self.num_shifts = num_shifts
            self.num_days = num_days
            self.cpmodel = model
            self.shifts = shifts

        def solve(self):
            solver = cp_model.CpSolver()
            solver.parameters.linearization_level = 0
            # Enumerate all solutions.
            solver.parameters.enumerate_all_solutions = True

            solution_limit = 5
            solution_printer = NursesPartialSolutionPrinter(
                self.shifts,
                self.num_nurses,
                self.num_days,
                self.num_shifts,
                solution_limit,
            )

            solver.solve(self.cpmodel, solution_printer)

            # Statistics.
            print("\nStatistics")
            print(f"  - conflicts      : {solver.num_conflicts}")
            print(f"  - branches       : {solver.num_branches}")
            print(f"  - wall time      : {solver.wall_time} s")
            print(f"  - solutions found: {solution_printer.solutionCount()}")
    return (Model,)


@app.cell
def _(Model):
    model = Model(num_nurses=4, num_shifts=3, num_days=6)
    model.solve()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
