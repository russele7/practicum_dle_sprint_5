# torch custom loss uncertainty weighting Kendall et al sigma example
# BaseModelOutputWithPoolingAndCrossAttentions

import torch
import torch.nn as nn
import torch.nn.functional as F

class KendallLossWeighting(nn.Module):
    """
    Implements the uncertainty-based loss weighting from Kendall et al. 2018
    for multi-task learning.
    """
    def __init__(self, num_tasks):
        super(KendallLossWeighting, self).__init__()
        # Initialize log(sigma^2) for each task as a learnable parameter
        # using zeros as a starting point.
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        Calculates the weighted loss sum.

        Args:
            losses (list of Tensors): List of individual task losses.
        
        Returns:
            Tensor: The combined, uncertainty-weighted loss.
        """
        assert len(losses) == len(self.log_sigma_sq), "Number of losses must match number of tasks"

        total_loss = 0
        for i, loss in enumerate(losses):
            # Formula from the paper: L_weighted = (1 / (2 * sigma_i^2)) * L_i + log(sigma_i)
            # We work with log_sigma_sq for numerical stability.
            
            # precision = 1 / (2 * sigma_i^2) = (1/2) * exp(-log_sigma_sq)
            precision = 0.5 * torch.exp(-self.log_sigma_sq[i])
            # log(sigma_i) = 0.5 * log(sigma_i^2) = 0.5 * log_sigma_sq
            log_sigma = 0.5 * self.log_sigma_sq[i]
            
            total_loss += precision * loss + log_sigma
            
        return total_loss



# Example for a model with 2 output heads (e.g., segmentation and depth estimation)
num_tasks = 2
loss_weighter = KendallLossWeighting(num_tasks)

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': loss_weighter.parameters(), 'weight_decay': 0} # Often no weight decay on sigma
], lr=...)


for epoch in range(num_epochs):
    for data in data_loader:
        optimizer.zero_grad()

        # Forward pass (model returns predictions for all tasks)
        pred_task1, pred_task2 = model(data)

        # Calculate individual losses using standard PyTorch criteria
        loss1 = criterion_task1(pred_task1, target_task1)
        loss2 = criterion_task2(pred_task2, target_task2)

        # Apply uncertainty weighting
        total_weighted_loss = loss_weighter(losses=[loss1, loss2])

        # Backward pass
        total_weighted_loss.backward()
        optimizer.step()
