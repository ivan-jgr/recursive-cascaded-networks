import torch


def pearson_correlation(fixed, warped):
    flatten_fixed = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(warped, start_dim=1)

    mean1 = torch.mean(flatten_fixed)
    mean2 = torch.mean(flatten_warped)
    var1 = torch.mean((flatten_fixed - mean1) ** 2)
    var2 = torch.mean((flatten_warped - mean2) ** 2)

    cov12 = torch.mean((flatten_fixed - mean1) * (flatten_warped - mean2))
    eps = 1e-6
    pearson_r = cov12 / torch.sqrt((var1 + eps) * (var2 + eps))

    raw_loss = 1 - pearson_r

    return raw_loss


def regularize_loss(flow):
    """
    flow has shape (batch, 2, 521, 512)
    """
    dx = (flow[..., 1:, :] - flow[..., :-1, :]) ** 2
    dy = (flow[..., 1:] - flow[..., :-1]) ** 2

    d = torch.mean(dx) + torch.mean(dy)

    return d / 2.0


def dice_loss(fixed_mask, warped):
    """
    Dice similirity loss
    """

    epsilon = 1e-6

    flat_mask = torch.flatten(fixed_mask, start_dim=1)
    flat_warp = torch.abs(torch.flatten(warped, start_dim=1))
    intersection = torch.sum(flat_mask * flat_warp)
    denominator = torch.sum(flat_mask) + torch.sum(flat_warp) + epsilon
    dice = (2.0 * intersection + epsilon) / denominator

    return 1 - dice


def jacobian_det(flow):
    raise NotImplementedError


def total_loss(fixed, moving, flows):
    sim_loss = pearson_correlation(fixed, moving)
    # Regularize all flows
    reg_loss = sum([regularize_loss(flow) for flow in flows])
    return sim_loss + reg_loss

