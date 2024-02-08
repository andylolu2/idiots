# %%
import torch
import torch.func as ft
from torch import nn

from idiots.utils import n_unsqueeze, tree_flatten, tree_map


# %%
class NN(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(1, width)
        self.fc2 = nn.Linear(width, 4)

    @property
    def out_ndims(self):
        return 1

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def f_true(x):
    return x**2


def single(f):
    def g(theta, *args):
        return f(theta, *(a.unsqueeze(0) for a in args)).squeeze(0)

    return g


x1 = torch.linspace(-1, 1, 10).reshape(-1, 1)
x2 = torch.linspace(-1, 1, 5).reshape(-1, 1)
# y = f_true(x)


# %%
def empirical_ntk(model):
    def K(x1, x2):
        """(*in) x (*in) -> (*out, *out)"""
        df_dtheta = ft.jacrev(ft.functional_call, argnums=1)
        theta = dict(model.named_parameters())
        jac1 = df_dtheta(model, theta, x1)
        jac2 = df_dtheta(model, theta, x2)

        def prod(j1, j2):
            """(*out, *param) x (*out, *param) -> (*out, *out)"""
            j1 = torch.flatten(j1, start_dim=model.out_ndims)
            j2 = torch.flatten(j2, start_dim=model.out_ndims)
            j1 = n_unsqueeze(j1, dim=model.out_ndims, n=model.out_ndims)
            j2 = n_unsqueeze(j2, dim=0, n=model.out_ndims)
            return torch.einsum("...k,...k->...", j1, j2)

        k = tree_map(prod, jac1, jac2)
        k = torch.stack(tree_flatten(k)[0]).sum(0)
        return k

    return K


# %%
model = NN(100)

K = empirical_ntk(model)
K = ft.vmap(ft.vmap(K, (0, None)), (None, 0))
print(K(x1, x2).shape)

# %%
