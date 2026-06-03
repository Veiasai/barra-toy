"""
Step-by-step ordinary least squares closed-form solution.

This file intentionally avoids NumPy so every matrix step is visible.

Model:
    y ~= a + b * x

Matrix form:
    y ~= X beta

where:
    X row = [1, x_i]
    beta = [a, b]'

Closed-form solution:
    beta_hat = (X'X)^(-1) X'y

Run:
    python least_squares_closed_form_step.py
"""

from __future__ import annotations


def print_matrix(name: str, matrix: list[list[float]]) -> None:
    print(f"\n{name}:")
    for row in matrix:
        print("  [" + ", ".join(f"{v:10.6f}" for v in row) + "]")


def print_vector(name: str, vector: list[float]) -> None:
    print(f"\n{name}:")
    for v in vector:
        print(f"  [{v:10.6f}]")


def transpose(a: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(a[0])
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    inner = len(a[0])
    cols = len(b[0])

    result = []
    for i in range(rows):
        out_row = []
        for j in range(cols):
            value = 0.0
            for k in range(inner):
                value += a[i][k] * b[k][j]
            out_row.append(value)
        result.append(out_row)

    return result


def matvec(a: list[list[float]], x: list[float]) -> list[float]:
    result = []
    for row in a:
        value = 0.0
        for a_ij, x_j in zip(row, x):
            value += a_ij * x_j
        result.append(value)
    return result


def inverse_2x2(a: list[list[float]]) -> list[list[float]]:
    """
    Inverse of:
        [a, b
         c, d]

    is:
        1 / (ad - bc) * [ d, -b
                         -c,  a]
    """
    a11 = a[0][0]
    a12 = a[0][1]
    a21 = a[1][0]
    a22 = a[1][1]

    det = a11 * a22 - a12 * a21
    if det == 0:
        raise ValueError("Matrix is singular; inverse does not exist.")

    return [
        [a22 / det, -a12 / det],
        [-a21 / det, a11 / det],
    ]


def subtract(a: list[float], b: list[float]) -> list[float]:
    return [a_i - b_i for a_i, b_i in zip(a, b)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


def main() -> None:
    # Four observed data points. We want to fit y ~= a + b*x.
    points = [
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 5.0),
        (4.0, 4.0),
    ]

    # Build the design matrix X.
    # The first column is 1 because the model has an intercept a.
    # The second column is x_i because the model has slope b.
    x_matrix = [[1.0, x] for x, _ in points]
    y_vector = [y for _, y in points]

    print("Goal: fit y ~= a + b*x")
    print("Unknown beta = [a, b]'")
    print("Closed form: beta_hat = (X'X)^(-1) X'y")

    print_matrix("Step 1 - X design matrix", x_matrix)
    print_vector("Step 1 - y observed values", y_vector)

    # Step 2: compute X'.
    x_transpose = transpose(x_matrix)
    print_matrix("Step 2 - X' transpose of X", x_transpose)

    # Step 3: compute X'X.
    # This summarizes how the explanatory variables overlap with each other.
    xtx = matmul(x_transpose, x_matrix)
    print_matrix("Step 3 - X'X", xtx)

    # Step 4: compute X'y.
    # This summarizes how the explanatory variables relate to observed y.
    y_as_column = [[v] for v in y_vector]
    xty_as_column = matmul(x_transpose, y_as_column)
    xty = [row[0] for row in xty_as_column]
    print_vector("Step 4 - X'y", xty)

    # Step 5: invert X'X.
    # Because this example has two parameters, X'X is 2x2.
    xtx_inv = inverse_2x2(xtx)
    print_matrix("Step 5 - (X'X)^(-1)", xtx_inv)

    # Step 6: beta_hat = (X'X)^(-1) X'y.
    beta_as_column = matmul(xtx_inv, xty_as_column)
    beta = [row[0] for row in beta_as_column]
    print_vector("Step 6 - beta_hat = [a, b]'", beta)

    intercept = beta[0]
    slope = beta[1]
    print(f"\nFitted line: y_hat = {intercept:.6f} + {slope:.6f} * x")

    # Step 7: compute predictions and residuals.
    y_hat = matvec(x_matrix, beta)
    residuals = subtract(y_vector, y_hat)
    squared_errors = [e * e for e in residuals]
    sse = dot(residuals, residuals)

    print_vector("Step 7 - y_hat = X beta_hat", y_hat)
    print_vector("Step 7 - residuals = y - y_hat", residuals)
    print_vector("Step 7 - squared residuals", squared_errors)
    print(f"\nStep 7 - sum of squared errors: {sse:.6f}")

    print("\nInterpretation:")
    print("  The chosen beta makes the sum of squared residuals as small as possible.")
    print("  If you changed a or b slightly, the squared-error sum would increase.")


if __name__ == "__main__":
    main()
