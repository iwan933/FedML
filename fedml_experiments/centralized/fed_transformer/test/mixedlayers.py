from typing import List

import torch
from torch import nn
from torch.nn import Parameter


class MixHiddenPresentations(nn.Module):
    def __init__(self, layer_num):
        super(MixHiddenPresentations, self).__init__()
        self.initial_scalar_parameters = [1.0] * layer_num
        self.scalar_parameters = nn.ParameterList(
            [
                Parameter(
                    torch.FloatTensor([self.initial_scalar_parameters[i]]), requires_grad=True
                )
                for i in range(layer_num)
            ]
        )
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, tensors: List[torch.Tensor]):
        print(self.initial_scalar_parameters)
        print("self.scalar_parameters: " + str(self.scalar_parameters))
        print("self.len: " + str(len(self.scalar_parameters)))
        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0
        )
        print("self.normed_weights: " + str(normed_weights))
        mixed_representations = []
        for weight, tensor in zip(normed_weights, tensors):
            mixed_representations.append(weight * tensor)
        return self.gamma * sum(mixed_representations)


model = MixHiddenPresentations(3)
tensor1 = torch.rand((2, 3))
print(tensor1)
tensor2 = torch.rand((2, 3))
print(tensor2)
tensor3 = torch.rand((2, 3))
print(tensor3)

out = model([tensor1, tensor2, tensor3])
print(out)
loss = sum(sum(out))
loss.backward()

for name, p in model.named_parameters():
    print(name)
    print(p.grad)

