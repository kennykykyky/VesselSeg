import numpy as np

def makeOpenA(alpha, beta, N):
    Alpha = [alpha]*N
    Beta = [beta]*N
    Beta[0] = 0
    Beta[-1] = 0

    A = np.zeros([N,N])

    for i in range(N):
        iplus1 = i + 1
        iplus2 = i + 2
        iminus1 = i - 1
        iminus2 = i - 2
        if iminus1 <= -1:
            iminus1 = -1 * iminus1
        if iminus2 <= -1:
            iminus2 = -1 * iminus2
        if iplus1 > N-1:
            iplus1 = 2 * (N - 1) - iplus1
        if iplus2 > N-1:
            iplus2 = 2 * (N - 1) - iplus2
        A[i,iminus2] = A[i,iminus2]+ Beta[iminus1]
        A[i, iminus1] = A[i, iminus1] - 2 * Beta[iminus1] - 2 * Beta[i] - Alpha[i]
        A[i, i] = A[i, i] + Alpha[i] + Alpha[iplus1] + 4 * Beta[i] + Beta[iminus1] + Beta[iplus1]
        A[i, iplus1] = A[i, iplus1] - Alpha[iplus1] - 2 * Beta[i] - 2 * Beta[iplus1]
        A[i, iplus2] = A[i, iplus2] + Beta[iplus1]

    return A

def norm_density(x,mu,sigma):
    epsilion = 1.192e-7#np.finfo(float).eps
    p = 1 / max(sigma * np.sqrt(2 * np.pi), epsilion) * np.exp(-pow(x - mu, 2) / max(2 * pow(sigma, 2), epsilion))
    return p
