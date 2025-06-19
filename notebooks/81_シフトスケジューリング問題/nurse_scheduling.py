# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
