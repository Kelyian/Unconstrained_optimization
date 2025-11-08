import numpy as np 
def gauss_newton(residuals,jacobian_func,x0,tol = 1e-6,max_iter = 1000):
    x = x0
    for k in range (max_iter):
        r = residuals(x)  #compute the residuals
        J = jacobian_func(x)  #compute the Jacobian matrix
        
        #solving for p = -(JTJ)^-1 (JTr)
        JTJ = J.T @ J
        JTr = J.T @ r
        try:
            p = -np.linalg.inv(JTJ) @ JTr
        except np.linalg.LinAlgError:
            print("Jacobian is a singular iteration",k)
            break
        
        x_new = x + p #update the parameters
        
        if np.linalg.norm(p) < tol:  #check for convergence
            print(f"Converged in {k} iterations.")
            x = x_new
            break
        
        x = x_new #moving to the next iteration
        
    print("Maximum iterations reached.")
    return x 
   
#the if main function to test the gauss newton method
if __name__ == "__main__":
    def residuals(x):
        x1,x2 = x
        r1 = 10*(x2 - x1**2)
        r2 = 1 - x2**2
        return np.array([r1,r2])
    
    def jacobian_func(X):
        x1,x2 = X
        J = np.array([[-20*x1,10],[0,-2*x2]])
        return J
    
    x0 = np.array([1.0,2.0]) #initial guess
    solution = gauss_newton(residuals,jacobian_func,x0)
    
    print("Optimal solution found:",solution)