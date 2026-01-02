# torch joint model custom loss example

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Define the joint model (example)
class JointModel(nn.Module):
    def __init__(self):
        super(JointModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc_cls = nn.Linear(50, 2) # Classification output (2 classes)
        self.fc_reg = nn.Linear(50, 1) # Regression output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        cls_out = self.fc_cls(x)
        reg_out = self.fc_reg(x)
        return cls_out, reg_out

# 2. Define the custom joint loss function
class JointLoss(nn.Module):
    def __init__(self, cls_weight=1.0, reg_weight=1.0):
        super(JointLoss, self).__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.MSELoss()

    def forward(self, predictions, targets):
        cls_preds, reg_preds = predictions
        cls_targets, reg_targets = targets

        loss_cls = self.cls_loss_fn(cls_preds, cls_targets)
        loss_reg = self.reg_loss_fn(reg_preds, reg_targets)

        # Combine losses with specific weights
        total_loss = self.cls_weight * loss_cls + self.reg_weight * loss_reg
        return total_loss

# 3. Example usage
if __name__ == '__main__':
    # Initialize model, loss, and optimizer
    model = JointModel()
    custom_criterion = JointLoss(cls_weight=0.5, reg_weight=1.5) # Emphasize regression loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Dummy data (batch size 4)
    inputs = torch.randn(4, 10)
    cls_targets = torch.randint(0, 2, (4,))
    reg_targets = torch.randn(4, 1)

    # Training step simulation
    optimizer.zero_grad()
    cls_preds, reg_preds = model(inputs)
    
    # Calculate the joint loss
    loss = custom_criterion((cls_preds, reg_preds), (cls_targets, reg_targets))
    
    print(f"Total Loss: {loss.item()}")

    # Backpropagate and update weights
    loss.backward()
    optimizer.step()
