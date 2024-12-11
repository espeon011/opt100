# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "ortools==9.11.4210",
# ]
# ///

import marimo

__generated_with = "0.9.34"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# 充足可能性問題と重み付き制約充足問題""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 重み付き制約充足問題""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        - 変数 $x_j \in D_j$ ($j = 1, \dots, n$) (各 $D_j$ は有限集合)
        - 制約 $C_i \subset D_1 \times \dots \times D_n$ ($i = 1, \dots, m$)

        制約充足問題とは, 上記条件を満たす $(x_1, \dots, x_n)$ を 1 組見つける(か解がないことを証明するか)問題のこと. 

        制約からの逸脱量 $g_i(x)$ が定義できればこれを最小にする問題としても定式化できる. 
        各制約の重みを $w_i$ として重み付き制約充足問題は

        \begin{align}
        &\text{minimize} \quad \sum_{i=1}^m w_i g_i(x) \\
        &\text{s.t.} \quad x_j \in D_j \quad \text{(for $j = 1, \dots, n$)}
        \end{align}

        最適化ソルバー SCOP が紹介されているが, 時間があればやる...
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 時間割作成問題""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### パラメータ

        - 授業 $i$
        - 教室 $k$
        - 教員 $l$
        - 学生
        - 期 $t$: ここでは 1 週間(5 日もしくは 6 日）の各時限を考える. 

        ### 時間割を組む条件

        - すべての授業をいずれか期へ割当
        - すべての授業をいずれか教室へ割当
        - 各期, 1 つの教室では 1 つ以下の授業
        - 同じ教員の受持ち講義は異なる期へ
        - 割当教室は受講学生数以上の容量をもつ
        - 同じ学生が受ける可能性がある授業の集合(カリキュラム)は, 異なる期へ割り当てなければならない

        ### 付加条件

        - 1 日の最後の期に割り当てられると履修学生数分のペナルティ
        - 1 人の学生が履修する授業が 3 連続するとペナルティ
        - 1 人の学生の授業数が 1 日に 1 つならばペナルティ(0 か 2 以上が望ましい)

        ### 定式化

        #### 変数

        - $x_{it}$: 授業 $i$ を期 $t$ に割り当てるとき $1$, そうでないとき $0$
        - $y_{ik}$: 授業 $i$ を教室 $k$ に割り当てるとき $1$, そうでないとき $0$

        #### 制約

        - すべての授業をいずれか期へ割当
          - $\sum_{t} x_{it} = 1$ (for all $i$)
        - すべての授業をいずれか教室へ割当
          - $\sum_{k} y_{ik} = 1$ (for all $i$)
        - 各期 $t$ の各教室 $k$ への割り当て授業は $1$ 以下
          - $\sum_{i} x_{it} y_{ik} \leq 1$ (for all $t, k$) ※ AND 制約なので線形制約で表現可能
        - 同じ教員の受持ち講義は異なる期へ割り当て
          - 教員 $l$ の受け持ち講義の集合を $E_l$ とする
          - $\sum_{i \in E_l} x_{it} \leq 1$ (for all $l, t$)
        - 割当教室は受講学生数以上の容量をもつ
          - 授業 $i$ ができない(容量を超過する)教室の集合を $K_i$ とする
          - $\sum_i \sum_{k \in K_i} y_{ik} \leq 0$
        - 同じ学生が受ける可能性がある授業の集合(カリキュラム)は異なる期へ割り当てなければならない
          - カリキュラム $j$ に含まれる授業の集合を $C_j$ とする
          - $\sum_{i \in C_j} x_{it} \leq 1$ (for all $j, t$)
        - 1 日の最後の期に割り当てられると履修学生数分のペナルティ
          - 1日の最後の期の集合を $L$, 授業 $i$ の履修学生数を $w_i$ とする
          - $\sum_{i} \sum_{t \in L} w_i x_{it} \leq 0$
        - 1 人の学生が履修する授業が 3 連続すると 1 ペナルティ
          - $T$ を 1 日のうちで最後の 2 時間でない期の集合とする
          - $\sum_{i \in C_j} (x_{i, t} + x_{i, t+1} + x_{i, t+2}) \leq 2$ (for all $t \in T$)
        - 1 人の学生の授業数が 1 日に 1 つならば 1 ペナルティ(0 か 2 以上が望ましい)
          - 各日 $d$ に含まれる期の集合を $T_d$ とする
          - 日 $d$ におけるカリキュラム $j$ に含まれる授業数が $0$ か $2$ 以上なのかを表す 0-1 変数 $z_{jd}$ とする
          - $\sum_{t \in T_d} \sum_{i \in C_j} x_{it} \leq |T_d| z_{jd}$ (for all $d, j$)
          - $\sum_{t \in T_d} \sum_{i \in C_j} x_{it} \geq 2 z_{jd}$ (for all $d, j$)
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## OR-tools (cp-sat)""")
    return


@app.cell
def __():
    from ortools.sat.python import cp_model
    import ortools

    ortools.__version__
    return cp_model, ortools


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 簡単な例""")
    return


@app.cell
def __(cp_model):
    _model = cp_model.CpModel()
    _var_upper_bound = max(50, 45, 37)
    _x = _model.new_int_var(0, _var_upper_bound, "x")
    _y = _model.new_int_var(0, _var_upper_bound, "y")
    _z = _model.new_int_var(0, _var_upper_bound, "z")
    _model.add(2 * _x + 7 * _y + 3 * _z <= 50)
    _model.add(3 * _x - 5 * _y + 7 * _z <= 45)
    _model.add(5 * _x + 2 * _y - 6 * _z <= 37)
    _model.maximize(2 * _x + 2 * _y + 3 * _z)
    _solver = cp_model.CpSolver()
    _status = _solver.solve(_model)

    if _status == cp_model.OPTIMAL:
        print(f"Maximum of objective function: {_solver.objective_value}")
        print()
        print(f"x value: {_solver.value(_x)}")
        print(f"y value: {_solver.value(_y)}")
        print(f"z value: {_solver.value(_z)}")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 全ての解の列挙""")
    return


@app.cell
def __(cp_model):
    class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
        """Print intermediate solutions."""

        def __init__(self, variables):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.__variables = variables
            self.__solution_count = 0

        def on_solution_callback(self):
            self.__solution_count = self.__solution_count + 1
            for v in self.__variables:
                print('%s=%i' % (v, self.value(v)), end=' ')
            print()

        @property
        def solution_count(self):
            return self.__solution_count
    return (VarArraySolutionPrinter,)


@app.cell
def __(VarArraySolutionPrinter, cp_model):
    # Creates the model.
    _model = cp_model.CpModel()

    # Creates the variables.
    _num_vals = 3
    _x = _model.new_int_var(0, _num_vals - 1, 'x')
    _y = _model.new_int_var(0, _num_vals - 1, 'y')
    _z = _model.new_int_var(0, _num_vals - 1, 'z')

    # Create the constraints.
    _model.add(_x != _y)

    # Create a solver and solve.
    _solver = cp_model.CpSolver()
    _solution_printer = VarArraySolutionPrinter([_x, _y, _z])
    _solver.parameters.enumerate_all_solutions = True
    _status = _solver.solve(_model, _solution_printer)

    print('Status = %s' % _solver.StatusName(_status))
    print(f'Number of solutions found: {_solution_printer.solution_count}')
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 例題: 覆面算""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        各文字に $0$ から $9$ の数字を入れて等式 $SEND+MORE=MONEY$ を成立させたい. 
        ただし, 数字に重複があってはならず先頭の文字に $0$ を入れることはできない.
        """
    )
    return


@app.cell
def __(cp_model):
    model = cp_model.CpModel()
    S = model.new_int_var(1, 9, 'S')
    E = model.new_int_var(0, 9, 'E')
    N = model.new_int_var(0, 9, 'N')
    D = model.new_int_var(0, 9, 'D')
    M = model.new_int_var(1, 9, 'M')
    O = model.new_int_var(0, 9, 'O')
    R = model.new_int_var(0, 9, 'R')
    Y = model.new_int_var(0, 9, 'Y')
    model.add(1000 * S + 100 * E + 10 * N + D + 1000 * M + 100 * O + 10 * R + E == 10000 * M + 1000 * O + 100 * N + 10 * E + Y)
    model.add_all_different([S, E, N, D, M, O, R, Y])
    _solver = cp_model.CpSolver()
    _status = _solver.solve(model)
    if _status == cp_model.OPTIMAL:
        for _v in [S, E, N, D, M, O, R, Y]:
            print(f'{_v} = {_solver.value(_v)}')
    return D, E, M, N, O, R, S, Y, model


@app.cell
def __(D, E, M, N, O, R, S, VarArraySolutionPrinter, Y, cp_model, model):
    _solution_printer = VarArraySolutionPrinter([S, E, N, D, M, O, R, Y])
    _solver = cp_model.CpSolver()
    _solver.parameters.enumerate_all_solutions = True
    _status = _solver.solve(model, _solution_printer)
    print(f"Status = {_solver.status_name(_status)}")
    print(f"Number of solutions found: {_solution_printer.solution_count}")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### 練習問題: 覆面算

        あとでやろう...
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 例題: 魔方陣""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        魔方陣とは $n \times n$の正方形の方陣に $1$ から $n^2$ までの整数を1つずつ入れて縦・横・対角線のいずれの列の和も同じになるものをいう. 
        (一列の和は $n (n^2 + 1) / 2$ となる)
        """
    )
    return


@app.cell
def __(cp_model):
    _n = 12 # n = 12 より大きい値で試さないこと(別に試しても問題はない)
    _model = cp_model.CpModel()
    _x = {}
    for _i in range(_n):
        for _j in range(_n):
            _x[_i, _j] = _model.new_int_var(1, _n * _n, f'x({_i},{_j})')
    _x_list = [_x[_i, _j] for _i in range(_n) for _j in range(_n)]
    _model.add_all_different(_x_list)
    _s = _n * (_n ** 2 + 1) // 2
    for _i in range(_n):
        _model.add(sum([_x[_i, _j] for _j in range(_n)]) == _s)
    for _j in range(_n):
        _model.add(sum([_x[_i, _j] for _i in range(_n)]) == _s)
    _model.add(sum([_x[_i, _i] for _i in range(_n)]) == _s)
    _model.add(sum([_x[_i, _n - _i - 1] for _i in range(_n)]) == _s)
    _solver = cp_model.CpSolver()
    _status = _solver.solve(_model)
    for _i in range(_n):
        for _j in range(_n):
            print(_solver.value(_x[_i, _j]), end=' ')
        print()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### 練習問題: 完全方陣

        あとでやろう...
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 例題: 数独""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        数独は $9 \times 9$ の正方形の枠内に $1$ から $n$ までの数字を入れるパズルである. 
        初期配置に与えられた数字はそのままとし, 縦・横の各列とブロックとよばれる $3 \times 3$ の小正方形の枠内には同じ数字を重複して入れてはいけないものとする. 

        以下のデータでは, 数字の $0$ が入っている枠は空白であるとする.
        """
    )
    return


@app.cell
def __(cp_model):
    import math
    _problem = [
        [1, 0, 0, 0, 0, 7, 0, 9, 0],
        [0, 3, 0, 0, 2, 0, 0, 0, 8],
        [0, 0, 9, 6, 0, 0, 5, 0, 0],
        [0, 0, 5, 3, 0, 0, 9, 0, 0],
        [0, 1, 0, 0, 8, 0, 0, 0, 2],
        [6, 0, 0, 0, 0, 4, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 7, 0, 0, 0, 3, 0, 0]
    ]
    model_sudoku = cp_model.CpModel()
    _n = len(_problem)
    _cell_size = math.ceil(math.sqrt(_n))
    _line_size = _cell_size ** 2
    line = range(0, _line_size)
    _cell = range(0, _cell_size)
    x = {}
    for _i in line:
        for _j in line:
            x[_i, _j] = model_sudoku.new_int_var(1, _line_size, f'x({_i},{_j})')
    for _i in line:
        model_sudoku.add_all_different([x[_i, _j] for _j in line])
    for _j in line:
        model_sudoku.add_all_different([x[_i, _j] for _i in line])
    for _i in _cell:
        for _j in _cell:
            _one_cell = []
            for _di in _cell:
                for _dj in _cell:
                    _one_cell.append(x[_i * _cell_size + _di, _j * _cell_size + _dj])
            model_sudoku.add_all_different(_one_cell)
    for _i in line:
        for _j in line:
            if _problem[_i][_j]:
                model_sudoku.add(x[_i, _j] == _problem[_i][_j])
    _solver = cp_model.CpSolver()
    _status = _solver.solve(model_sudoku)
    for _i in line:
        for _j in line:
            print(_solver.value(x[_i, _j]), end=' ')
        print()
    return line, math, model_sudoku, x


@app.cell
def __(VarArraySolutionPrinter, cp_model, line, model_sudoku, x):
    _solver = cp_model.CpSolver()
    _solution_printer = VarArraySolutionPrinter([x[_i, _j] for _i in line for _j in line])
    _solver.parameters.enumerate_all_solutions = True
    _status = _solver.solve(model_sudoku, _solution_printer)
    print('Status = %s' % _solver.status_name(_status))
    print('Number of solutions found: %i' % _solution_printer.solution_count)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### 練習問題: 数独

        気が向いたらやる...
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### 練習問題: 不等式パズル

        やっておきたい
        """
    )
    return


if __name__ == "__main__":
    app.run()
