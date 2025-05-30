from scipy.io import mmread
from IterativeMethods import IterativeMethods 
from Util import Util
import numpy as np
import itertools

def main():
    spd_cond = False
    dd_cond = False

    files = ["spa1.mtx","spa2.mtx","vem1.mtx","vem2.mtx"]
    methods = ["Jacobi","GaussSeidel","Gradient","ConjugateGradient"]
    tolerances = [1e-4,1e-6,1e-8,1e-10]
    results = {}
    util = Util()

    for file in files:
        A = util.readFromFile(file)
        print("Matrix loaded from " + file)
        xexact = np.ones(A.shape[0])
        b = np.dot(A, xexact)
        x0 = np.zeros(A.shape[0])
        it = IterativeMethods(A,b,xexact,x0)

        spd_cond = util.spdCheck(A)
        dd_cond = util.isDiagonallyDominant(A)
        print("----------Condizioni-----------")
        if(dd_cond):
            print(f"La matrice {file} è a dominanza diagonale, Jacobi e Gauss-Seidel convergono!")
        else:
            print(f"La matrice {file} non è a dominanza diagonale, Jacobi e Gauss-Seidel possono non convergere!")


        if(spd_cond):
            print(f"La matrice {file} è SPD, gradiente e il gradiente coniugato convergono!")
        else:
            print(f"La matrice {file} non è SPD, gradiente può non convergere, gradiente coniugato non applicabile!")

        for toll in tolerances:
            print(f"-----------Tolerance: {toll}-----------")
            results[f'Jacobi_{toll}'] = it.jacobi(toll)
            results[f'GaussSeidel_{toll}'] = it.Gauss_Seidel(toll)
            results[f'Gradient_{toll}'] = it.gradient(toll)
            if(spd_cond):
                results[f'ConjugateGradient_{toll}'] = it.CG(toll)
    
        it.groupPlotResult(results,tolerances,methods,file)


main()