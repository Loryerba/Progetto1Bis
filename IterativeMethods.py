import time
import numpy as np
import matplotlib.pyplot as plt
from Util import Util

class IterativeMethods:

    def __init__(self, A, b, xe, x0, maxIter=20000):
        """
        Create an object for resolving linear systems in the form Ax = b.

        Parameters
        ----------
        A: np.ndarray
            SPD Matrix in np format representing Ax = b linear systems.
        b: np.ndarray constant terms
        xe: np.ndarray
            Exact system solutions.
        x0: np.ndarray
            Initial guess array used to start the first iteration.
        maxIter: int,optional
            Iteration ends if number of iteration is < maxIter.

        Returns
        -------
        IterativeMethods instance object.
        """
        self.A = A
        self.b = b
        self.xe = xe
        if(maxIter < 20000 ):
            self.maxIter = 20000
        else:
            self.maxIter = maxIter
        self.x0 = x0

    def __triangLower(self, L, b):
        """
        Solve lower triangular systems in the form Lx = b.

        Parameters
        ----------
        L: np.ndarray
            Matrix as np.ndarray object.

        Returns
        -------
        output : np.ndarray
            Exact system solution.
        """
        M, N = L.shape
        x = np.zeros(M)

        x[0] = b[0] / L[0, 0]
        for i in range(1, N):
            x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
        
        return x
    
    def __getRelativeError(self,x_app):
        return np.linalg.norm(x_app-self.xe) / np.linalg.norm(self.xe)


    def __getErrorForExit(self,xnew):
        return np.linalg.norm(np.dot(self.A, xnew) - self.b) / np.linalg.norm(self.b)

    def jacobi(self, toll, info = True):
        """
        Execute Jacobi iterative method.

        Parameters
        ----------
        info: bool, optional
            If True let function prints the approximate result, error, 
            number of iterations and running time. Default is True.

        Returns
        -------
        output : tuple, bool
            Returns tuple object like (xnew,err,nit,time_elapsed) representing
            approximate result, error, number of iterations and running time, if D is invertible,
            False otherwise.
        """
        util = Util()
        D = np.diag(np.diagonal(self.A))
        if(util.diagonalNotZero(D) is False):
            print("[WARN] La diagonale non è invertibile!")
            return False
        B = D - self.A  # B = D - A
        D_inv = 1 / np.diagonal(D) #D^-1 = 1.0 / D

        xold = self.x0
        err = 1
        nit = 0
        stime = time.time()
        while err > toll and nit < self.maxIter:
            xnew = D_inv * (np.dot(B, xold) + self.b)   # xnew = D^-1(Bx +b)
            err = self.__getErrorForExit(xnew) 
            xold = xnew
            nit = nit + 1

        
        elapsed_time = (time.time() - stime) * 1000

        err = self.__getRelativeError(xnew)
        if(nit == self.maxIter):
            print("Convergenza non raggiunta!")
        
        if info:
            print("Jacobi")
            print("\tNumero di iterazioni:", nit)
            print("\tErrore:", err)
            print("\tTempo di esecuzione:", elapsed_time, "ms\n")

        return xnew, err ,nit, elapsed_time

    def Gauss_Seidel(self,toll, info=True):
        """
        Execute Gauss-Seidel iterative method.

        Parameters
        ----------
        info: bool, optional
            If True let function prints the approximate result, error, 
            number of iterations and running time. Default is True.

        Returns
        -------
        output : tuple
            Returns tuple object like (xnew,err,nit,time_elapsed) representing
            approximate result, error, number of iterations and running time.
        """
        D_L = np.tril(self.A) # (D-L) = triang(A)
        U = self.A - D_L        # U = A - (D-L)
        xold = self.x0
        nit = 0
        err=1
        stime = time.time()
        while(err > toll and nit < self.maxIter):
            xnew = self.__triangLower(D_L,(self.b-np.dot(U,xold)))
            err = self.__getErrorForExit(xnew)
            xold = xnew
            nit = nit+1
        

        elapsed_time = (time.time()-stime)*1000


        err = self.__getRelativeError(xnew)

        if(nit == self.maxIter):
            print("Convergenza non raggiunta!")

        if info:
            print("Gauss_Seidel")
            print("\tNumero di iterazioni:", nit)
            print("\tErrore:", err)
            print("\tTempo di esecuzione:", elapsed_time, "ms\n")

        return xnew,err,nit,elapsed_time


    def gradient(self, toll, info = True):
        """
        Execute gradient iterative method.

        Parameters
        ----------
        info: bool, optional
            If True let function prints the approximate result, error, 
            number of iterations and running time. Default is True.

        Returns
        -------
        output : tuple
            Returns tuple object like (xnew,err,nit,time_elapsed) representing
            approximate result, error, number of iterations and running time.
        """
        nit=0
        xold=self.x0
        err = 1
        stime = time.time()
        while(err > toll and nit < self.maxIter):
            residual = self.b - np.dot(self.A,xold)     # r = b - Ax
            alpha = np.dot(residual.T,residual)/np.dot(residual.T,np.dot(self.A,residual))  #alpha = (r*r^T) / (r^T*A*r)
            xnew = xold + alpha*residual        #xnew = xold + alpha *r
            err = self.__getErrorForExit(xnew)
            xold = xnew
            nit = nit+1
        
        elapsed_time = (time.time()-stime)*1000

        err = self.__getRelativeError(xnew)

        if(nit == self.maxIter):
            print("Convergenza non raggiunta!")

        if info:
            print("Gradiente")
            print("\tNumero di iterazioni:", nit)
            print("\tErrore:", err)
            print("\tTempo di esecuzione:", elapsed_time, "ms\n")
        return xnew,err,nit,elapsed_time

    def CG(self, toll, info = True):
        """
        Execute conjugate gradient iterative method.

        Parameters
        ----------
        info: bool, optional
            If True let function prints the approximate result, error, 
            number of iterations and running time. Default is True.

        Returns
        -------
        output : tuple,bool
            Returns tuple object like (xnew,err,nit,time_elapsed) representing
            approximate result, error, number of iterations and running time, if matrix is SPD.
            False otherwise
        """
        util = Util()
        if(util.spdCheck(self.A) is False):
            print("[WARN] La matrice in input non è SPD!")
            return False
        rold = self.b - (np.dot(self.A,self.x0))           # r^(k) = b - A x^(k)
        dold = rold.copy()              # d^(k) = r^(k)
        err = 1
        nit = 0
        xold = self.x0
        stime = time.time()
        while err > toll and nit < self.maxIter:
            y_k = (np.dot(self.A,dold))           # y^(k) = A d^(k)
            alpha = (dold @ rold) / (dold @ y_k)   # a_k

            xnew = xold + alpha * dold           # x^(k+1)
            rnew = rold - alpha * y_k           # r^(k+1)

            w_k = np.dot(self.A,rnew)            # w^(k) = A r^(k+1)
            beta_k = (dold @ w_k) / (dold @ y_k)   # b_k
            dnew = rnew - beta_k * dold          # d^(k+1)

            err = self.__getErrorForExit(xnew)  
            xold = xnew
            rold = rnew
            dold = dnew
            nit += 1

        elapsed_time = (time.time() - stime)*1000


        err = self.__getRelativeError(xnew)

        if(nit == self.maxIter):
            print("Convergenza non raggiunta!")

        if info:
            print("Gradiente coniugato")
            print("\tNumero di iterazioni:", nit)
            print("\tErrore:", err)
            print("\tTempo di esecuzione:", elapsed_time, "ms\n")
        return xnew, err, nit, elapsed_time
    

    def singlePlotResult(self, method, err_list, elapsed_time_list, iterations_list,tol_list,file):
        """
        Generate a single plot for a specific method.

        Parameters
        ----------
        method: string
            name of executed method
        err_list: list
            list of error collected during the execution
        elapsed_time_list: list
            list of elapsed time collected during the execution
        iterations_list:
            list of number of iterations collected during the execution
        tol_list: list
            list of tolerances
        file: string
            name of file
        """
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        ax1.plot(tol_list, err_list, marker='d', label=method)
        ax2.plot(tol_list, elapsed_time_list, marker='d', label=method)
        ax3.plot(tol_list, iterations_list, marker='d', label=method)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Tolerance')
        ax1.set_ylabel('Error')
        ax1.set_title(f'Tolerance vs Error for {file}')
        if ax1.has_data():
            ax1.legend()
        ax1.grid(True)

        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Tolerance')
        ax2.set_ylabel('Execution Time (ms)')
        ax2.set_title(f'Tolerance vs Execution Time for {file}')
        if ax2.has_data():
            ax2.legend()
        ax2.grid(True)

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Tolerance')
        ax3.set_ylabel('Iterations')
        ax3.set_title(f'Tolerance vs Iterations for {file}')
        if ax3.has_data():
            ax3.legend()
        ax3.grid(True)

        plt.subplots_adjust(hspace=0.7)
        plt.show()    
    

    def groupPlotResult(self,results,tol_list,methods,file):
        """
        Generate an aggregate plot for different methods.

        Parameters
        ----------
        results: dict
            dictionary containing the results of the methods
        tol_list: list
            list of tolerances
        methods: list
            list of methods
        file: string
            name of file
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        

        for m in methods:
            error_list = []
            time_list = []
            nit_list = []
            for tol in tol_list:
                key = f"{m}_{tol}"

                try:
                    xnew, err, nit, elapsed_time = results[key]
                except KeyError:
                    print(f"[WARN] Chiave mancante: {key}")

                time_list.append(elapsed_time)
                error_list.append(err)
                nit_list.append(nit)
            
            ax1.plot(tol_list, error_list, marker='d', label=m)
            ax2.plot(tol_list, time_list, marker='d', label=m)
            ax3.plot(tol_list, nit_list, marker='d', label=m)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Tolerance')
        ax1.set_ylabel('Error')
        ax1.set_title(f'Tolerance vs Error for {file}')
        if ax1.has_data():
            ax1.legend()
        ax1.grid(True)

        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Tolerance')
        ax2.set_ylabel('Execution Time (ms)')
        ax2.set_title(f'Tolerance vs Execution Time for {file}')
        if ax2.has_data():
            ax2.legend()
        ax2.grid(True)

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Tolerance')
        ax3.set_ylabel('Iterations')
        ax3.set_title(f'Tolerance vs Iterations for {file}')
        if ax3.has_data():
            ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(f"Plot_{file}.png")
        plt.show() 