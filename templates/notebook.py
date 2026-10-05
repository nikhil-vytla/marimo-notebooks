# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.25.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Title

    Replace with a one-line description of the notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
