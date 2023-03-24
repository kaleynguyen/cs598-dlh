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
w = np.zeros_like(X, dtype=np.float32)

def forward(x):
      return w * x
def loss(y, y_hat):
      return ((y - y_hat)**2).mean()
def gradient(x, y, y_hat):
      return np.dot(2*x, y_hat-y).mean()
print(f'Prediction before training: {forward(X)}')
learning_rate = 0.01
n_iters = 5
for epoch in range(n_iters):
      # prediction: forward pass
      y_pred = forward(X)
      # loss
      l = loss(Y, y_pred)
      # gradient
      dw = gradient(X, Y, y_pred)
      # update weights
      w -= learning_rate * dw
print(f'Prediction after training: {forward(X)}')
```


# Backpropagation through Pytorch
```python
import torch
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
	return w * x
def loss(y, y_hat):
	return ((y-y_hat)**2).mean()
print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# Start training
learning_rate = 0.01
n_iters = 100
for epoch in range(n_iters):
      # prediction: forward pass
      y_pred = forward(X)
      # loss
      l = loss(Y, y_pred)
      # final gradient of dl/dw by backward pass through local gradients
      l.backward() #dl/dw
      # update weights
      with torch.no_grad():
      	w -= learning_rate * w.grad
      #zero gradients
      w.grad.zero_()
      if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')
```
# Training pipeline
1. Design model (input, output, forward pass)
2. Construct loss and optimizer
3. Training loop
	- forward pass: compute prediction
	- backward pass: compute gradients
	- update weights 
```python
import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
n_samples, n_features = X.shape


class LinearRegression(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(LinearRegression, self).__init__()
		self.lin = nn.Linear(input_dim, output_dim)
	def forward(self, x):
		return self.lin(x)

model = nn.Linear(n_features, n_features)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print(f'Prediction before training: f(5) = {model(torch.tensor([5], dtype=torch.float32)).item():.3f}')

# Start training

n_iters = 200
for epoch in range(n_iters):
      # prediction: forward pass
    y_pred = model(X)
      # loss
    l = loss(Y, y_pred)
      # final gradient of dl/dw by backward pass through local gradients
    l.backward() #dl/dw
      # update weights
    optimizer.step()
      #zero gradients
    optimizer.zero_grad()
    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {model(torch.tensor([5], dtype=torch.float32)).item():.3f}')
```


