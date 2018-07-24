import torch
import numpy as np

# Tensors are similar to NumPy’s ndarrays.

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)

print(np_data)
print(torch_data)

x1 = torch.empty(5,3)
print(x1)