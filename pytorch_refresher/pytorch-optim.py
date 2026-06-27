import math, os, torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# use the optim package to define an optimizer
# first arg tells what the opt should update
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
loss_hist = []
for t in range(2000):
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    loss_hist.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# make a plot showing the loss curve
os.makedirs('pytorch_refresher/data/pytorch-optim', exist_ok=True)
plt.figure(figsize=(12, 5))
plt.plot(loss_hist)
plt.title("Loss curve")
plt.savefig('pytorch_refresher/data/pytorch-optim/loss_curve.png')