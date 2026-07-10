import torch
import torch.nn.functional as F


class CustomMSECorrLoss(torch.nn.Module):
    def __init__(self, loss_weights={"mse": 0.5, "temporal": 0.5, "roi": 0.0}):
        """
        :param loss_weights: relative weights of the loss terms
            - "mse": classic MSE loss (mean squared error between prediction and target)
            - "temporal": Pearson-correlation loss along the temporal dimension for each ROI
            - "roi": Pearson-correlation loss across ROIs (per sample)
        """
        super().__init__()
        self.loss_weights = loss_weights  # e.g. {"mse": 1.0, "temporal": 0.5, "roi": 0.0}

    def forward(self, y_pred, y_true):
        """
        :param y_pred: (batch_size, num_roi) predicted fMRI ROI values
        :param y_true: (batch_size, num_roi) ground-truth fMRI ROI values
        :return: the combined loss
        """
        total_loss = 0.0

        # MSE term
        if self.loss_weights["mse"] > 0:
            mse_loss = F.mse_loss(y_pred, y_true)
            total_loss += self.loss_weights["mse"] * mse_loss

        # Temporal correlation loss (correlation along the temporal/batch dimension per ROI)
        if self.loss_weights["temporal"] > 0:
            y_pred_mean = y_pred.mean(dim=0, keepdim=True)  # (1, num_roi)
            y_true_mean = y_true.mean(dim=0, keepdim=True)

            y_pred_centered = y_pred - y_pred_mean
            y_true_centered = y_true - y_true_mean

            numerator = (y_pred_centered * y_true_centered).sum(dim=0)
            denominator = torch.sqrt((y_pred_centered ** 2).sum(dim=0)) * torch.sqrt((y_true_centered ** 2).sum(dim=0))

            temporal_correlation = numerator / (denominator + 1e-8)  # per-ROI temporal correlation
            temporal_correlation_loss = 1 - temporal_correlation.mean()  # average over ROIs

            total_loss += self.loss_weights["temporal"] * temporal_correlation_loss

        # ROI-wise correlation loss (correlation across ROIs, per sample)
        if self.loss_weights["roi"] > 0:
            batch_size, num_roi = y_pred.shape  # (batch, num_roi)

            batch_correlations = []
            for i in range(batch_size):
                # single-sample ROI vectors
                pred_i = y_pred[i].unsqueeze(0)  # (1, num_roi)
                true_i = y_true[i].unsqueeze(0)  # (1, num_roi)

                # stack (2, num_roi) and compute the Pearson correlation
                combined = torch.cat([pred_i, true_i], dim=0)  # (2, num_roi)
                correlation_matrix = torch.corrcoef(combined)  # (2, 2)

                correlation = correlation_matrix[0, 1]  # Pearson correlation between pred and true
                batch_correlations.append(correlation)

            # average Pearson-correlation loss over the batch
            roi_correlation_loss = 1 - torch.stack(batch_correlations).mean()

            total_loss += self.loss_weights["roi"] * roi_correlation_loss

        return total_loss
