import torch, os, math
import matplotlib.pyplot as plt
import pandas as pd

'''
autograd is jsut a caller for the forward and backward passes.
we can create our own subclass of torch.autograd.Function and implement the forward and backward passes ourselves, and autograd will call it appropriately.

will use legendre polynomials 
'''

class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod
    def forward(input):
         return 0.5 + (5 * input ** 3 - 3 * input)

    @staticmethod
    def setup_context(ctx, inputs, output):
        '''
        store inpuit for use ion the backward pass
        '''
        input, = inputs
        ctx.save_for_backward(input)
        
    @staticmethod
    def backward(ctx, grad_output):
        '''
        we will receive a tensor with the fradient of the loss wrt output
        need to compute the gradient of the loss wrt input and return it
        '''
        
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)
    
dtype = torch.float
device = torch.device("cpu")
torch.set_default_dtype(dtype)

x =torch.linspace(-math.pi, math.pi, 2000, device=device)
y = torch.sin(x)

# create random tensors for weights, initialized to random values
# cant be too far from the solution or it will diverge, so we initialize them to small random values

a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)
        
learning_rate = 5e-6
losses = []
for t in range(2000):
    P3 = LegendrePolynomial3.apply
    
    y_pred = a + b * P3(c + d * x)
    
    loss = (y_pred - y).pow(2).sum()
    losses.append(loss.item())
    if t % 100 == 99:
        print(f'Iteration {t}, loss = {loss.item():.4f}')
    
    # use autograd to compute the backward pass
    loss.backward()
    
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        
        a.grad.zero_()
        b.grad.zero_()
        c.grad.zero_()
        d.grad.zero_()
    
print(f'Result: y = {a.item()} + {b.item()} P3({c.item()} + {d.item()} x)')

# plot the loss curve and the final fit against the data and the true function
os.makedirs('pytorch_refresher/data/custom-autograd', exist_ok=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# make loss curve log scale to see the convergence better
plt.yscale('log')
plt.plot(losses)
plt.title("Loss curve")
plt.subplot(1, 2, 2)
plt.scatter(x.cpu(), y.cpu(), label="Data")
# make log scale for y axis to see the fit better
plt.plot(x.cpu(), y_pred.detach().cpu(), label="Fit", color="red")
plt.plot(x.cpu(), torch.sin(x).cpu(), label="True function", color="green")
plt.legend()
plt.title("Cubic fit to sin(x)")
plt.savefig('pytorch_refresher/data/custom-autograd/legendre_fit.png')