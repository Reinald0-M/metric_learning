import math
import numpy as np
import os
# create random data
x = np.linspace(-math.pi, math.pi, 20000)
y = np.sin(x)

# trying to fit a cubic function like a + b x + c x^2 + d x^3

# randomly init weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-12
losses = []
for t in range(2000):
    # forward pass: compute predicted y
    y_pred = a + b  * x + c * x **2 + d * x **3
    
    # compute loss
    loss = np.square(y_pred - y).sum()
    losses.append(loss)
    if t%100 == 99:
        print(t, loss)
        
    # backprop to compute grad of a,b,c,d wrt loss
    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()
    
    # update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
    
print(f"Result: y = {a} + {b} x + {c} x^2 + {d} x^3")

# plot the loss curve and the final fit against the data and the true function
os.makedirs('pytorch_refresher/data/warmup', exist_ok=True)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss curve")
plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, label="Fit", color="red")
plt.plot(x, np.sin(x), label="True function", color="green")
plt.legend()
plt.title("Cubic fit to sin(x)")
plt.savefig('pytorch_refresher/data/warmup/cubic_fit.png')
