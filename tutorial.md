# Intro
```python
import torch
x = torch.empty(1)
x = torch.rand(3,3)
x = torch.zeros(2,3)
x = torch.ones(2,5, dtype=torch.int8)
print(x.dtype)
print(x.size())

# Basic operations
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x * y #element-wise operation 
y.add_(x) #inplace operation

#Slicing
x = torch.rand(5,3)
x[:,0] #first row torch.Size([5])
x[0,:] #first col torch.Size([3])
x[:, :1] #first row of 2d tensor, torch.Size([5, 1])
x[:1, :] #first col of 2d tensor, torch.Size([1, 3])
x[1,1].item() #second row second col actual value access

#Reshape
x = torch.rand(3,4)
x.view(4,3)
x.view(-1,6)
```


# Autograd
```python
import torch
x = torch.randn(3, requires_grad=True)
y = x + 2
z = 2 * y * y
z.backward(torch.tensor([0.01, 0.01, 0.02], dtype=torch.float32))
print(x.grad)
x.requires_grad_(False) # or True
```

# Backpropagation

```python

```

# Intro
```python
import torch
x = torch.empty(1)
x = torch.rand(3,3)
x = torch.zeros(2,3)
x = torch.ones(2,5, dtype=torch.int8)
print(x.dtype)
print(x.size())

# Basic operations
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x * y #element-wise operation 
y.add_(x) #inplace operation

#Slicing
x = torch.rand(5,3)
x[:,0] #first row torch.Size([5])
x[0,:] #first col torch.Size([3])
x[:, :1] #first row of 2d tensor, torch.Size([5, 1])
x[:1, :] #first col of 2d tensor, torch.Size([1, 3])
x[1,1].item() #second row second col actual value access

#Reshape
x = torch.rand(3,4)
x.view(4,3)
x.view(-1,6)
```


# Autograd
```python
import torch
x = torch.randn(3, requires_grad=True)
y = x + 2
z = 2 * y * y
z.backward(torch.tensor([0.01, 0.01, 0.02], dtype=torch.float32))
print(x.grad)
x.requires_grad_(False) # or True
```

# Backpropagation (backward pass of automatic gradients at each node so the loss can be propagated back to the parameters W => minimize the loss) and linear regression
* $$\hat{y} = w^Tx$$ `w.T @ x`
* $$Loss = (\hat{y} - y)^2 = (w^Tx - y)^2$$

```python
import torch
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)
y_hat = w*x
loss = (y_hat - y)**2
loss.backward()
w.grad #-2
```

# Backpropagation through Numpy
```python
import numpy as np
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

def forward(x):
      return w * x
def loss(y, y_hat):
      return ((y - y_hat)**2).mean()
def gradient(x, y, y_hat):
      return np.dot(2*x, y_hat-y).mean()
print(f'Prediction before training: {forward(5): 0.3f}')
learning_rate = 0.01
n_iters = 20
for epoch in range(n_iters):
      # prediction: forward pass
      y_pred = forward(X)
      # loss
      l = loss(Y, y_pred)
      # gradient
      dw = gradient(X, Y, y_pred)
      # update weights
      w -= learning_rate * dw
      if epoch % 2 == 0:
            print('epoch {0}: w = {1:0.3f}, loss = {2:0.5f}'.format(epoch+1, w, l))
print(f'Prediction after training: {forward(5): 0.3f}')
```



