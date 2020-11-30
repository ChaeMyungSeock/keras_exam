import numpy as np
#
# t = np.array([0., 1., 2., 3., 4., 5., 6.])
# # 파이썬으로 설명하면 List를 생성해서 np.array로 1차원 array로 변환함.
# print(t)
#
# print('Rank of t: ', t.ndim)
# print('Shape of t: ', t.shape)
#
#
import torch
# t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
# print(t)
#
# print(t.dim())  # rank. 즉, 차원
# print(t.shape)  # shape
# print(t.size()) # shape
#
# m1 = torch.FloatTensor([[1, 2], [3, 4]])
# m2 = torch.FloatTensor([[1], [2]])
# print('Shape of Matrix 1: ', m1.shape) # 2 x 2
# print('Shape of Matrix 2: ', m2.shape) # 2 x 1
# print(m1.matmul(m2)) # 2 x 1

ft = torch.Tensor([0, 1, 2])
print(ft.shape)
print(ft.unsqueeze(1)) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
print(ft.unsqueeze(1).shape)