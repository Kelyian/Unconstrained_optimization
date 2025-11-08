import numpy as np
def newtons_method(f,grad_f,hess_f,x0,tol=1e-6,max_iter=500):
    #initilaizing the starting point
    x= x0.astype(float)
    for k in range(max_iter):
        #computing gradient at the current point
        grad = grad_f(x)
        #computing hessian at the current point
        hess = hess_f(x)
        #checking if the hessian is invertible (if gradient = 0 --> stationary point)
        if np.linalg.norm(grad)<tol:
            break
        #computing the search direction
        p = -np.linalg.solve(hess,grad) # better than inverting the hessian
        #after getting the ssearch direction we update the new point
        x = x + p
    return x,f(x),k + 1
#testing the codes functionality
if __name__ == "__main__":
    #setting the function
    def f(x):
        x1,x2 = x
        return ([x1**2 + 2*x2**2 - 3*x1 - 2*x2])
    #gradient
    def grad_f(x):
        x1,x2 = x
        return np.array([ 2*x1 - 3 , 4*x2 - 2])
    #hession matrix
    def hess_f(x):
        return np.array([[2,0],
                         [0,4]])
    x0 = np.array([2,1])
    x_opt,f_opt,n_iter = newtons_method(f,grad_f,hess_f,x0)
    print("Using Newton's Method for Optimization")
    print("\n")
    print("optimal point:",x_opt)
    print("optimal value:",f_opt)
    print("convergences after ",n_iter," iterations:")
    
    