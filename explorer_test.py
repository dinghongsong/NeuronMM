import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import neuronxcc as nx
# XLA imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp

import torch_neuronx
# from torch_neuronx.experimental import profiler

# Global constants
EPOCHS = 2

# Declare 3-layer MLP Model
class MLP(nn.Module):
  def __init__(self, input_size = 10, output_size = 2, layers = [5, 5]):
      super(MLP, self).__init__()
      self.fc1 = nn.Linear(input_size, layers[0])
      self.fc2 = nn.Linear(layers[0], layers[1])
      self.fc3 = nn.Linear(layers[1], output_size)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return F.log_softmax(x, dim=1)


def main():
    # Fix the random number generator seeds for reproducibility
    torch.manual_seed(0)

    # XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
    device = xm.xla_device()

    # Start the proflier context-manager
    with torch_neuronx.profiler.profile(
        profile_type='operator',
        target='neuron_profile',
        output_dir='./output') as profiler:

        # IMPORTANT: the model has to be transferred to XLA within
        # the context manager, otherwise profiling won't work
        model = MLP().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.NLLLoss()

        # start training loop
        print('----------Training ---------------')
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            train_x = torch.randn(1,10).to(device)
            train_label = torch.tensor([1]).to(device)

            #forward
            loss = loss_fn(model(train_x), train_label)

            #back
            loss.backward()
            optimizer.step()

            # XLA: collect ops and run them in XLA runtime
            xm.mark_step()

    print('----------End Training ---------------')

if __name__ == '__main__':
    main()