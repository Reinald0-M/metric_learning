import os, torch, math, random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))
        
    def forward(self, x):
        # randomly choose 4,5 and resuse e parameter to compute the contribution of these orders
        # since each forward pass builds a dynamic computation graph, we can randomly choose which orders to use and reuse the same parameter for different orders
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y += self.e * x ** exp
        return y
    
    def string (self):
        return f"y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4/5"
    

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = DynamicNet()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9])
loss_hist = []

for t in range(30000):
    y_pred = model(x)
    
    loss = criterion(y_pred, y)
    loss_hist.append(loss.item())
    if t % 1000 == 999:
        print(t, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# visualize the loss curve and the final fit against the data and the true function
os.makedirs('pytorch_refresher/data/pytorch-control-flow-weight-sharing', exist_ok=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_hist)
plt.xlabel('Iteration')
plt.yscale('log')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.subplot(1, 2, 2)
plt.scatter(x.detach().numpy(), y.detach().numpy(), label="Data")
plt.plot(x.detach().numpy(), y_pred.detach().numpy(), label="Fit", color="red")
plt.plot(x.detach().numpy(), np.sin(x.detach().numpy()), label="True function", color="green")
plt.legend()
plt.title(f"DynamicNet fit to sin(x): {model.string()}")
plt.savefig('pytorch_refresher/data/pytorch-control-flow-weight-sharing/dynamic_fit.png')