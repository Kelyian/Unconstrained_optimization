#------computing optimization using linear search method-------
import numpy as np
def steepest_descent(f,grad_f,x0,alpha = 1.0,rho=0.5,c=1e-4,tol=1e-6,max_iter = 500):
    x = x0.copy()
    #stores the optimization process for later review
    history = []
    #OPTIMIZATION LOOP
    for k in range(max_iter):
        #computing the gradient
        grad = grad_f(x)
        #compute the gradient norm
        grad_norm = np.linalg.norm(grad)
        #check for convergence
        if grad_norm < tol:
            print(f"Converged in {k} iterations.")
            break
        #compute the search direction
        p = -grad
        #initializing the step size
        alpha0 = alpha
        #backtracking line search
        while f(x + alpha*p) > f(x) +c*alpha*np.dot(grad,p):
            alpha *= rho  #reduces the step size
        #move to the next point along the search direction with the determined step size
        x = x + alpha*p
        #store information for analysis
        history.append((k,x.copy() ,f(x), grad_norm, alpha))
    x_opt = x
    f_opt = f(x_opt)
    return x_opt, f_opt, history
#the if main function to test the linear search method
if __name__ == "__main__":
    def f(x):
        #the test equation
        return (x[0]-x[1])+2*(x[0])*(x[1]) + 2*(x[0])**2 + (x[1])**2
    def grad_f(x):
        #the gradient of the test equation
        return np.array([1 + 2*(x[1]) + 4*(x[0]), -1 + 2*(x[0]) + 2*(x[1])])
    #starting point 
    x0 = np.array([0.0,0.0])
    #running the linear search optimization
    x_opt, f_opt, history = steepest_descent(f,grad_f,x0)
    print("Optimal solution:", x_opt)
    print("Optimal value:", f_opt)
    
    