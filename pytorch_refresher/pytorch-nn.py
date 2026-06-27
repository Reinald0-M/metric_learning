import torch, math, os
import matplotlib.pyplot as plt


# create tensors to hold inputts and outputs
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# make output a linear function of x, x^2, x^3
p = torch.tensor([1,2,3])
xx = x.unsqueeze(-1).pow(p)

# use nn opackage to define model as sequence of layser. 
# nn.Sequential is a kind of container that allows us to build a neural network as a sequence of layers.

model = torch.nn.Sequential(
    # Linear layer with 3 inputs and 1 output, followed by a flatten layer to convert the output to a 1D tensor
    torch.nn.Linear(3,1),
    torch.nn.Flatten(0,1)
)

# mean squared error loss
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6

a_hist, b_hist, c_hist, d_hist, loss_hist = [], [], [], [], []
for t in range(2000):
    # forward pass: compute prediucted y by giving x to the model
    y_pred = model(xx)
    
    # compute the loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    loss_hist.append(loss.item())
    # zero the gradients before running the backward pass
    model.zero_grad()
    
    loss.backward()
    
    # backward pass; compute the loss gradient for all learnable parameters
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
# access the first layer of the model like a list
    linear_layer = model[0]
    current_params = torch.cat([linear_layer.weight.flatten(), linear_layer.bias])
    with torch.no_grad():
        weights = model[0].weight.flatten().tolist() # a, b, c
        bias = model[0].bias.item()                  # d
        
        a_hist.append(weights[0])
        b_hist.append(weights[1])
        c_hist.append(weights[2])
        d_hist.append(bias)


os.makedirs('pytorch_refresher/data/pytorch-nn', exist_ok=True)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.plot3D(a_hist, c_hist, loss_hist, 'blue', label='Optimization Path')
ax.scatter3D(a_hist[0], c_hist[0], loss_hist[0], color='green', s=100, label='Start')
ax.scatter3D(a_hist[-1], c_hist[-1], loss_hist[-1], color='red', s=100, label='End')

ax.set_xlabel('Weight a ($x$)')
ax.set_ylabel('Weight c ($x^3$)')
ax.set_zlabel('Loss')
ax.set_title('3D Parameter Space Trajectory')
plt.legend()
plt.savefig('pytorch_refresher/data/pytorch-nn/parameter_space_trajectory.png')

