#%%
import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"

#%%
arr = np.load(section_dir / "numbers.npy")
print(arr[0].shape)
display_array_as_img(arr[0])

#%%
three_marked = arr[3].copy()
three_marked[0:2, 0:10, 0:10] = 0
three_marked[1, 0:10, -10:] = 255
three_marked[2, -10:, 0:10] = 0
three_marked[0:2, -10:, -10:] = 200
display_array_as_img(three_marked)
three_transposed = einops.rearrange(three_marked, "c h w -> c w h")
display_array_as_img(three_transposed)

# %%
horizontal_stack = einops.rearrange(arr, "b c h w -> c h (b w)")
display_array_as_img(horizontal_stack)

# %%
### - SKIPPING BASIC EINOPS FUNCTIONS FOR NOW

# %% BROADCASTING
a = t.tensor([1, 2, 3])
b = t.tensor([[1], [2], [3]])
print(a.shape, b.shape)
c = a + b
print(c.shape)
print(c)
# %%
a = t.tensor([1, 2, 3])
b = t.tensor([1, 2, 3, 4])
c = a + b # This errors! Broadcasting cannot do this.
print(c.shape)
print(c)
# %%
