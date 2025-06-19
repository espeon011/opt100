# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "ortools==9.13.4784",
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
    mo.md(r"""# 整数最適化問題""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align}
    &\text{maximize} & y + z \\
    &\text{s.t.} & x + y + z = 32 \\
    & & 2 x + 4 y + 8 z \leq 80 \\
    & & x, y, z \in \mathbb{Z}_{\geq 0}
    \end{align}
    """
    )
    return


@app.cell
def _():
    from ortools.math_opt.python import mathopt
    return (mathopt,)


@app.cell
def _(mathopt):
    model = mathopt.Model(name="getting_started_milp")
    x = model.add_integer_variable(lb=0, name="x")
    y = model.add_integer_variable(lb=0, name="y")
    z = model.add_integer_variable(lb=0, name="z")

    model.add_linear_constraint(x + y + z == 32)
    model.add_linear_constraint(2 * x + 4 * y + 8 * z <= 80)

    model.maximize(y + z)
    return model, x, y, z


@app.cell
def _(mathopt, model):
    params = mathopt.SolveParameters(enable_output=True)
    result = mathopt.solve(model, mathopt.SolverType.HIGHS, params=params)
    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        raise RuntimeError(f"model failed to solve: {result.termination}")
    return (result,)


@app.cell
def _(result, x, y, z):
    print(f"x = {result.variable_values()[x]}")
    print(f"y = {result.variable_values()[y]}")
    print(f"z = {result.variable_values()[z]}")
    print(f"objective = {result.objective_value()}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
