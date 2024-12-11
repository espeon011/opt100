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
    mo.md(r"""# 線形最適化問題""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        \begin{align}
        &\text{maximize} & 15x_1 + 18 x_2 & \\
        &\text{s.t.} & 2x_1 + x_2 &\leq 60 \\
        & & x_1 + 2 x_2 &\leq 60 \\
        & & x_1, x_2 &\geq 0
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
    model = mathopt.Model(name="getting_started_lp")

    x1 = model.add_variable(lb=0, name="x1")
    x2 = model.add_variable(lb=0, name="c2")

    model.add_linear_constraint(2 * x1 + x2 <= 60)
    model.add_linear_constraint(x1 + 2 * x2 <= 60)

    model.maximize(15 * x1 + 18 * x2)
    return model, x1, x2


@app.cell
def __(mathopt, model):
    params = mathopt.SolveParameters(enable_output=True)
    result = mathopt.solve(model, mathopt.SolverType.GLOP, params=params)
    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        raise RuntimeError(f"model failed to solve: {result.termination}")
    return params, result


@app.cell
def __(result, x1, x2):
    print(f"x1 = {result.variable_values()[x1]}")
    print(f"x2 = {result.variable_values()[x2]}")
    print(f"objective = {result.objective_value()}")
    return


if __name__ == "__main__":
    app.run()
