#%%
import torch as t
from einops import einsum

#%%
a = t.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

b = t.tensor([
    [2,2,2],
    [2,2,2],
    [2,2,2]
])

print(a.shape)
print(a)
print(b.shape)
print(b)

c = a @ b
print(c)
d = einsum(a, b, "i j, j k -> i k")
print(d)

# Broadcasting example
bias = t.tensor([1, 2, 3])
e = d + bias
print(e)

#%%
print("-----------------------------------\n\n")
residual_representation = t.tensor([
    [[1, 0], [2, 1], [3, 1], [4, 0]],
    [[6, 0], [5, 0], [4, 0], [3, 0]],
    [[2, 0], [2, 0], [2, 0], [2, 0]],
])
print(residual_representation.shape) # (batch, sequence, d_model) -> (3, 4, 2)

query_weights = t.tensor([
    [[2,2,2],[2,2,2]],
    [[1,1,1],[1,1,1]],
    [[2,2,2],[2,2,2]],
    [[1,1,1],[1,1,1]],
])
print(query_weights.shape) # (n_heads, d_model, d_head) -> (4, 2, 3)

queries_no_bias = einsum(residual_representation, query_weights, "batch sequence d_model, n_heads d_model d_head -> batch sequence n_heads d_head")

print(queries_no_bias.shape)
print(queries_no_bias)

# %%
