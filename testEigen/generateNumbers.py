import numpy as np
totalCount = 756
totalDims = 10000

a = np.random.rand(totalCount, totalDims)
b = np.random.rand(totalCount, totalDims)
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html#numpy.einsum
self = np.einsum('ij,ij->i', a, b)

np.savetxt("rand.csv", self, delimiter=",")

corr_ori = np.dot(a, b.T) # This should be cnt*cnt martix
corr = corr_ori.reshape(-1,1)
mx = max(corr)
mn = min(corr)
# We select 60% between max and min
thres = 0.4*mx + 0.6*mn
corr = corr[corr>thres]
corr = corr - thres + 1
np.savetxt("corr.csv", corr)

'''
Next for rich get richer, actually we simplify as 'communication'
As communication can be modeled as power of matrix multi,
and matrix multi can be calculated as SVD
so this is power of Eigenvalues
'''
eigen = np.linalg.eigvals(np.dot(a,a.T))
communications = 10
eigen = np.power(eigen, communications)
eigen = np.log(eigen - np.min(eigen) + 1)
#eigen = eigen[eigen>0]
print("eigen", eigen)
# remove largest eigen
eigen = eigen[1:]
np.savetxt("eigen.csv", eigen)
