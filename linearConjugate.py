import numpy as np
import matplotlib.pyplot as plt
# Function to extract A and b from a quadratic expression
# f(x) = 0.5 * x^T A x - b^T x
def extract_A_b(Q, c):
    A = np.array(Q, dtype=float)
    b = np.array(c, dtype=float)
    return A, b
# Define the quadratic function and gradient
def quadratic_function(x, A, b):
    return 0.5 * np.dot(x, A @ x) - np.dot(b, x)
def quadratic_gradient(x, A, b):
    return A @ x - b
# Nonlinear Conjugate Gradient (NCG) method for quadratic f(x)
def conjugate_gradient_quadratic(A, b, x0, tol=1e-8, max_iter=1000):
    x = np.array(x0, dtype=float)
    r = b - A @ x           # residual (negative gradient)
    p = r.copy()            # initial direction
    f_values = []
    print("Iteration | ||r||")
    print("------------------")
    for k in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        f_values.append(np.linalg.norm(r_new))
        print(f"{k:9d} | {np.linalg.norm(r_new):.3e}")
        # Check convergence
        if np.linalg.norm(r_new) < tol:
            print("\nConverged!")
            break
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        x, r = x_new, r_new
    # Final solution
    print(f"\nFinal solution: x = {x}")
    print(f"Residual norm = {np.linalg.norm(r_new):.3e}")
    return x
# Example: Quadratic function
# Suppose we have:
# f(x) = 0.5*(3x1² + 2x1x2 + 6x2²) - (2x1 + x2)
# Then A = [[3, 1], [1, 6]],  b = [2, 1]
Q = [[3, 1],
     [1, 6]]
c = [2, 1]
# Extract A and b
A, b = extract_A_b(Q, c)
# Initial guess
x0 = np.array([0.0, 0.0])
# Run Conjugate Gradient
solution = conjugate_gradient_quadratic(A, b, x0)
# Compare with analytical solution (A⁻¹b)
x_true = np.linalg.solve(A, b)
print(f"\nAnalytical solution: {x_true}")
