# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pyscipopt==5.4.1",
# ]
# ///

import marimo

__generated_with = "0.11.25"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 錐最適化問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        \begin{align}
        &\text{maximize} & 2 x + 2 y + z \\
        &\text{s.t.} & x^2 + y^2 \leq z^2 \\
        & & 2 x + 3 y + 4 z \leq 10 \\
        & & x, y, z \geq 0
        \end{align}
        """
    )
    return


@app.cell
def _():
    from pyscipopt import Model
    return (Model,)


@app.cell
def _(Model):
    model = Model()

    x = model.addVar(vtype='C', lb=0, ub=None, name='x')
    y = model.addVar(vtype='C', lb=0, ub=None, name='y')
    z = model.addVar(vtype='C', lb=0, ub=None, name='z')

    cons_lin = model.addCons(2 * x + 3 * y + 4 * z <= 10)
    cons_socp = model.addCons(x * x + y * y <= z * z)

    model.setObjective(2 * x + 2 * y + z, sense="maximize")

    model.optimize()
    return cons_lin, cons_socp, model, x, y, z


@app.cell
def _(model, x, y, z):
    x_val = model.getVal(x)
    y_val = model.getVal(y)
    z_val = model.getVal(z)

    print(f"x = {x_val}")
    print(f"y = {y_val}")
    print(f"z = {z_val}")
    print(f"objective = {model.getObjVal()}")

    # SCIP の連続変数の最適解は微妙に制約を破っていたりする
    print(f"x^2 + y^2 <= z^2 ?: {x_val ** 2 + y_val ** 2 <= z_val ** 2}")
    return x_val, y_val, z_val


if __name__ == "__main__":
    app.run()
