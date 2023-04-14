# Links

* [Lab solution] https://drive.google.com/drive/folders/1wRpt0L1YxUA9VIeVZbPMlNLcNIiVwYK_
* [project instruction] https://www.overleaf.com/project/63f3d5b99a8e992a6db47d13
* [Dive into deep learning] https://d2l.ai
* [citi training]
* [ml4drug] https://ml4drug-book.github.io
* [synthetic mimic] https://app.medisyn.ai
* [pyhealth] https://pyhealth.readthedocs.io/en/latest/index.html
* [eICU database request access]


# Lab 1

### basic

```

#basic
x = torch.tensor([[1,2], [3,4]])
x = torch.from_numpy(np.array([[1,2], [3,4]]))
x_ones = torch.ones_like(x) #x can be a tensor or a tuple of shape (1,2,)
x_rand = torch.rand_like(x)
x.shape, x.dtype, x.device

# tensor concatenation
torch.cat([tensor, tensor, tensor], dim=1) #concat row-wise so WIDE
torch.cat([tensor, tensor, tensor], dim=-2) #concat column-wise so TALL

# matrix multiplication
tensor @ tensor.T
# element-wise product
tensor * tensor
# convert to a numerical value 
x.sum().item()
```

### Exercise

* sigmoid: `return 1 / (1 + torch.exp(-x))`
* softmax: `return torch.exp(X) / torch.exp(X).sum(1, keepdim=True)`#normalized row-wise so each row sum to 1
* linear layer: `torch.matmul(X, W) + b`
* squared-loss: `((y_hat - y.reshape(y_hat.shape)) ** 2 / 2).mean()`
* cross_entropy: `l = -y * torch.log(Y_hat)`, `L = l.sum(dim=1).mean()`
* autograd: 

```
x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward() #same as dy
x.grad == 4*x
```

### Whole pipeline

```
def linear(X, W, b):
    return torch.matmul(X, W) + b

def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2 / 2).mean()

def sgd(params, lr, batch_size):
    """  Minibatch stochastic gradient descent """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

lr = 0.03
num_epochs = 20
net = linear
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

