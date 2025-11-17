import torch
from matplotlib import pyplot as plt

def fun(x, m, s):
    return ((x / m) ** 2) * torch.exp(-(x - m) / s)


# Example usage
x = torch.linspace(0, 10, 500)
y = fun(x, 1, 1)
print(y.shape)
plt.plot(x.numpy(), y.numpy())
plt.grid(True)
plt.legend()
plt.show()
