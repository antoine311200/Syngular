from syngular.variational.optimizer import Lanczos

import numpy as np

if __name__ == "__main__":

    n = 64
    m = 4
    A = np.zeros(n**2).reshape((n,n))
    np.fill_diagonal(A, list(range(n)))

    T, V = Lanczos.fit(A, iteration=m)
    print(T)
    print(V)

    eigenvaluesA, eigenvectorsA = np.linalg.eig(A)
    eigenvaluesT, eigenvectorsT = np.linalg.eig(T)

    import matplotlib.pyplot as plt

    plt.plot(eigenvaluesA, np.ones(n)*0.2,  '+' )
    plt.plot(eigenvaluesT, np.ones(m)*0.1,  '+' )
    plt.ylim(0,1)
    plt.show()