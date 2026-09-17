import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
torch.set_num_threads(1)
print('Threads set to 1')
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
print('Calling backward...')
y.backward()
print('Grad:', x.grad)
