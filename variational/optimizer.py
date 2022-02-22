from re import M
import numpy as np

class Optimizer:
    
    @staticmethod
    def fit(tensor):
        return tensor

class Lanczos(Optimizer):
    
    @staticmethod
    def fit(tensor, iteration=100):
        n = tensor.shape[0]*tensor.shape[1] if len(tensor.shape) > 2 else tensor.shape[0]
        m = min(n, iteration)

        A = np.reshape(tensor, newshape=(n, -1)) if len(tensor.shape) > 2 else tensor

        T = np.zeros((m, m))
        V = np.zeros((m, n))

        # First iteration

        v = np.random.rand(n)
        v /= np.linalg.norm(v)
        V[0, :] = v

        w = np.dot(A, v)
        alpha = np.dot(w, v)
        w = w - alpha*v

        T[0,0] = alpha

        for j in range(1, m-1):
            beta = np.sqrt(np.dot(w,w))

            vj = w/beta
            for i in range(j-1):
                vi = V[i, :]
                vj = vj - np.dot(np.conj(vj), vi)*vi
            vj = vj/np.linalg.norm(vj)

            w = np.dot(A, vj)
            alpha = np.dot(w, vj)
            w = w - alpha * vj - beta*V[j-1, :]

            V[j, :] = vj

            T[j,j] = alpha
            T[j-1,j] = beta
            T[j,j-1] = beta
        
        return T, V
