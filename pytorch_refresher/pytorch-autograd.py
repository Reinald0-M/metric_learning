import math
import os
import torch
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")
torch.set_default_dtype(dtype)

x = torch.linspace(-1,1, 2000, dtype=dtype)
y = torch.exp(x)

# create random tensors for weights, initialized to random values
a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)

initial_loss = 1.
learning_rate = 1e-5
losses = []
for t in range(5000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    
    # compute loss on tensors
    loss = (y_pred - y).pow(2).sum()
    losses.append(loss.item())
    
    if t==0:
        initial_loss = loss.item()
    if t % 100 == 99:
        print(f'Iteration {t}, loss(t)/loss(0) = {loss.item()/initial_loss:.4f}')
    
    # use autograd to compute the backward pass
    loss.backward()
     
     # manually update weights using gradient descent
     # wapp the code to use torch.no_grad() to prevent tracking history in autograd
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        
        # zero the gradients after updating to reset them for the next iteration
        a.grad.zero_()
        b.grad.zero_()
        c.grad.zero_()
        d.grad.zero_()
        
    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
    
    
# plot the loss curve and the final fit against the data and the true function
os.makedirs('pytorch_refresher/data/pytorch-autograd', exist_ok=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss curve")
plt.subplot(1, 2, 2)
plt.scatter(x.numpy(), y.numpy(), label="Data", s=0.5)
plt.plot(x.numpy(), y_pred.detach().numpy(), label="Fit", color="red")
plt.plot(x.numpy(), torch.exp(x).numpy(), label="True function", color="green")
plt.legend()
plt.title("Cubic fit to exp(x)")
plt.savefig('pytorch_refresher/data/pytorch-autograd/cubic_fit.png')