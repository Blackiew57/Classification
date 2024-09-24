from torch import nn, Tensor

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes:int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,num_classes),
        )
        
    def forward(self, x:Tensor) -> Tensor:
        
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits