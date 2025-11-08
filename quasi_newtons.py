#solving using Symmetric rank-one matrix updates
import numpy as np
#solving for a valid value of alpha 
def linesearch(f,grad_f,xk,pk,alpha0=1,c=1e-4,rho=0.5,max_iter=1000):
    alpha = alpha0
    fxk = f(xk)
    gdotp = np.dot(grad_f(xk),pk)
    #finding a valid alpha
    while f(xk +alpha * pk )>fxk +c*alpha*gdotp:
        alpha *= rho #reduce alpha
    return alpha
#solving the quasi-newton method function
def quasi_newton(f,grad_f,x0,N0 = None,tol=1e-6,max_iter=1000):
    #initializing variables
    xk = x0
    n = len(xk)
    Nk = np.eye(n) if N0 is None else N0 #initial inverse hessian approximation
    for k in range(max_iter):
        #compute gradient at this poimt
        gk = grad_f(xk)
        #check for convergence
        if np.linalg.norm(gk) < tol:
            print(f"converged in {k} iterations")
            break
        #compute search direction)(pk = -N_k * g_k)
        pk = -np.linalg.inv(Nk) @ gk
        #line search to find step size alpha
        alpha = linesearch(f,grad_f,xk,pk)
        #getting value f the new x
        x_next = xk + alpha * pk
        #compute S_K and Y_k
        sk = x_next - xk
        yk = grad_f(x_next) - gk
        #updating n_k 
        NkYk = Nk @ yk
        diff = (sk - NkYk)
        denom = diff.T @ yk 
        if abs(denom) > 1e-10: # this ensures the denomenator is not equal to zero
            Nk = Nk + (diff @ diff.T)/denom 
        #move to the next iteration
        xk = x_next      
    return xk,f(xk),k+1
#example
if __name__ == "__main__":
    def f(x):
        x1,x2 = x
        return (x1**2) - 2*x1*x2 + 4*x2**2
    def grad_f(x):
        x1,x2 = x
        df_dx1 = 2*x1 - 2*x2
        df_dx2 = -2*x1 + 8*x2
        return np.array([df_dx1,df_dx2])
    x0 = np.array([-3.0,1.0])
    x_opt,f_min,n_iter = quasi_newton(f,grad_f,x0)
    print("Using Quasi-Newton's Method for Optimization")
    print("\n")
    print("optimal point:",x_opt)
    print("optimal value:",f_min)
    print("convergences after ", n_iter," iterations:")      
    