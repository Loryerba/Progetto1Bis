import numpy as np
from scipy.io import mmread
from scipy.linalg import cholesky

class Util:

    def checkInstance(self,A):
        """
        Check if A is an instance of np.ndarray.

        Parameters
        ----------
        A: np.ndarray
            Matrix array-like.
        
        Returns
        -------
        output: bool
            True if A is an instance of np.ndarray, 
            False otherwise.
        
        """
        #check if A is instance of np.ndarray
        return isinstance(A,np.ndarray)

    def squareCheck(self,A):
        """
        Check if A is a square matrix.

        Parameters
        ----------
        A: np.ndarray
            Matrix array-like.
        
        Returns
        ------
        output: bool
            Returns True if A is square, False if A is not instance of np.ndarray or not square.
        """
        #number of rows == number of columns
        return A.shape[0] == A.shape[1] if self.checkInstance(A) is True else False
    
            
    def matrixVectorSameLength(self,A,b):
        """
        Check if b is a constant terms array with same size of A (rows or columns).

        Parameters
        ----------
        A: np.ndarray
            Matrix array-like.
        b: np.ndarray
            Array of constant terms.

        Returns
        -------
        output: bool
            Returns True if b is the same size of A. False if at least one is not
            instance of np.ndarray or not the same size.
        """
        #check if number of rows is same size of b 
        return A.shape[0] == b.shape[0] if (self.checkInstance(A) and self.checkInstance(b)) is True else False

    def diagonalNotZero(self,A):
        """
        Check if diagonal not contains zero elements.

        Parameters
        ----------
        A: np.ndarray
            Matrix array-like.
        
        Returns
        -------
        output: bool
            Returns True if every element on the diagonal is not zero, 
            False if A is not an instance of np.ndarray or has zero on diagonal.
        """
        #consider 0 values < trheshold
        threshold = 1e-16
        #check if any element in diagonal is < threshold
        return not np.any(np.abs(np.diag(A)) < threshold) if self.checkInstance(A) else False

    def readFromFile(self,path):
        """
        Read .mtx file and get np.ndarray.

        Parameters
        ----------
        path: string
            Path to the .mtx file.
        
        Returns
        -------
        output: np.ndarray
            Returns matrix as np.ndarray like.
        """
        # read the .mtx file
        return mmread(path).toarray()

    def symmetryCheck(self,A):
        """
        Check if matrix is symmetric.

        Parameters
        ----------
        A: np.ndarray
            Matrix array-like.
        
        Returns
        -------
        object: bool
            Returns True if A is symmetric,
            False otherwise.
        """
        # check if a A and the transposed are equal
        return np.allclose(A,A.T) if self.checkInstance(A) and self.squareCheck(A) else False

    def spdCheck(self,A):
        """
        Check if matrix is SymmetricPositiveDefinite.

        Parameters
        ----------
        A: np.ndarray
            Matrix array-like.
        
        Returns
        -------
        output: bool
            Returns True if A is symmetric and positive definite, 
            False otherwise.
        """
        # first check if A is symmetric
        if(self.symmetryCheck(A)):
            try:
                # run cholesky decomposition to check if A is positive definite
                cholesky(A)
                return True
            except:
                return False
        else:
            return False
    
    def isDiagonallyDominant(self,A):
        """
        Check if matrix is diagonally dominant

        Parameters
        ----------
        A: np.ndarray
            Matrix array-like.
        
        Returns
        -------
        output: bool
            Returns True if A is diagonally dominant, 
            False otherwise.
        """
        if(self.checkInstance(A)):
            for i, row in enumerate(A):
                s = sum(abs(v) for j, v in enumerate(row) if i != j)
                if s > abs(A[i][i]):
                    return False
            return True
        else:
            return False
        