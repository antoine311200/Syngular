from re import M
import numpy as np

class Optimizer:
    
    @staticmethod
    def fit(tensor):
        return tensor

class Lanczos(Optimizer):
    
    @staticmethod
    def fit(tensor, init_vec = None, iteration=100) -> tuple[np.ndarray]:
        n = tensor.shape[0]*tensor.shape[1] if len(tensor.shape) > 2 else tensor.shape[0]
        l = tensor.shape[2]*tensor.shape[3] if len(tensor.shape) > 2 else tensor.shape[1]
        m = min(n, iteration)

        A = np.reshape(tensor, newshape=(n, -1)) if len(tensor.shape) > 2 else tensor
        T = np.zeros((m, m))
        V = np.zeros((m, l))

        if A.shape[0] != A.shape[1]:
            A = np.dot(A.conj().T, A)

        # First iteration


        if init_vec is None:
            v = np.random.rand(l)
            v /= np.linalg.norm(v)
        else:
            v = init_vec[:l]
        V[0, :] = v

        w = np.dot(A, v)
        alpha = np.dot(w, v)
        w = w - alpha*v

        T[0,0] = alpha

        for j in range(1, m):
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


class Davidson:
    
    @staticmethod
    def fit(tensor, init_vec = None, iteration=100) -> tuple[np.ndarray]:
        pass