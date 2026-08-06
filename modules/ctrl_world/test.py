import math
from pprint import pprint

import torch

# input = torch.randn(3, 5, requires_grad=True)  # [3, 5]
# target = torch.randint(5, (3,), dtype=torch.int64)  # [3]
input = torch.tensor(
    [
        [1, 1, 9, 1, 1],
        [1, 15, 1, 1, 1],
        [1, 1, 1, 18, 1],
    ],
    dtype=torch.float32,
)
input = torch.nn.functional.softmax(input, dim=-1)
target = torch.tensor([2, 1, 3], dtype=torch.int64)
pprint(input.shape)
pprint(input)
pprint(target.shape)
pprint(target)
loss = torch.nn.functional.cross_entropy(input, target)
pprint(loss)
