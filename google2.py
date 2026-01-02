# pytorch multi task model custom loss example

import torch
import torch.nn as nn

# 1. Define the Multi-Task Model
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        # Shared layers (backbone)
        self.shared_fc = nn.Linear(10, 5)
        # Task 1 head (e.g., classification head)
        self.task1_fc = nn.Linear(5, 2)
        # Task 2 head (e.g., regression head)
        self.task2_fc = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.shared_fc(x))
        output1 = self.task1_fc(x)
        output2 = self.task2_fc(x)
        return output1, output2

# 2. Define individual loss functions
criterion1 = nn.CrossEntropyLoss() # For classification
criterion2 = nn.MSELoss()         # For regression

# 3. Implement the custom combined loss function
def multi_task_loss(output1, target1, output2, target2, weight1=1.0, weight2=0.5):
    loss1 = criterion1(output1, target1)
    loss2 = criterion2(output2, target2)
    # Combine losses with custom weights
    total_loss = weight1 * loss1 + weight2 * loss2
    return total_loss, loss1, loss2

# Example Usage within a training loop:
if __name__ == '__main__':
    # Initialize model, optimizer
    model = MultiTaskModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Dummy data
    inputs = torch.randn(3, 10)
    # Target 1 is class indices for CrossEntropyLoss
    target1 = torch.tensor([0, 1, 1])
    # Target 2 is continuous values for MSELoss
    target2 = torch.randn(3, 1).squeeze(1)

    # Forward pass
    output1, output2 = model(inputs)

    # Calculate the total loss using the custom function
    total_loss, loss1_val, loss2_val = multi_task_loss(output1, target1, output2, target2, weight1=1.0, weight2=0.5)

    print(f"Loss 1: {loss1_val.item():.4f}, Loss 2: {loss2_val.item():.4f}")
    print(f"Total Weighted Loss: {total_loss.item():.4f}")

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward() # Backpropagates through all combined losses
    optimizer.step()

