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
    mo.md(r"""# 整数最適化問題""")
    return


@app.cell(hide_code=True)
def __(mo):
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
def __():
    from ortools.math_opt.python import mathopt
    return (mathopt,)


@app.cell
def __(mathopt):
    model = mathopt.Model(name="getting_started_milp")
    x = model.add_integer_variable(lb=0, name="x")
    y = model.add_integer_variable(lb=0, name="y")
    z = model.add_integer_variable(lb=0, name="z")

    model.add_linear_constraint(x + y + z == 32)
    model.add_linear_constraint(2 * x + 4 * y + 8 * z <= 80)

    model.maximize(y + z)
    return model, x, y, z


@app.cell
def __(mathopt, model):
    params = mathopt.SolveParameters(enable_output=True)
    result = mathopt.solve(model, mathopt.SolverType.HIGHS, params=params)
    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        raise RuntimeError(f"model failed to solve: {result.termination}")
    return params, result


@app.cell
def __(result, x, y, z):
    print(f"x = {result.variable_values()[x]}")
    print(f"y = {result.variable_values()[y]}")
    print(f"z = {result.variable_values()[z]}")
    print(f"objective = {result.objective_value()}")
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
