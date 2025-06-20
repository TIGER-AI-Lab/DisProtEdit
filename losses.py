import torch


def lalign(x, y, alpha=2): 
    # bsz : batch size (number of positive pairs)
    # d : latent dim
    # x : Tensor, shape=[bsz, d]
    # latents for one side of positive pairs
    # y : Tensor, shape=[bsz, d]
    # latents for the other side of positive pairs
    # lam : hyperparameter balancing the two losses

    return (x - y).norm(dim=1).pow(alpha).mean() # scalar output

def lunif(x, t=2): 
    # bsz : batch size (number of positive pairs)
    # d : latent dim
    # x : Tensor, shape=[bsz, d]
    # latents for one side of positive pairs
    # y : Tensor, shape=[bsz, d]
    # latents for the other side of positive pairs
    # lam : hyperparameter balancing the two losses

    sq_pdist = torch.pdist(x, p=2).pow(2) # pairwase distances between all pairs of rows in X
    return sq_pdist.mul(-t).exp().mean().log() # aggregates the information from all pairwise, scalar output

def l_mmd(latent1, latent2, kernel='rbf', sigma=1.0):
    """Compute the MMD loss between two latent distributions
    latent1 and latent2 do not necessarily need to have the same dimensions.
    
    Args:
        latent1 (Tensor): Latent variable tensor of shape (batch_size, latent_dim1)
        latent2 (Tensor): Latent variable tensor of shape (batch_size, latent_dim2)
        kernel (str): Kernel type, 'rbf' or 'linear'. Default is 'rbf'.
        sigma (float): Bandwidth for the RBF kernel. Default is 1.0.
    
    Returns:
        Tensor: MMD loss value.
    """
    def rbf_kernel(x, y, sigma):
        pairwise_dists = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-pairwise_dists / (2 * sigma ** 2))

    def linear_kernel(x, y):
        return torch.mm(x, y.T)

    # Choose kernel function
    if kernel == 'rbf':
        kernel_func = lambda x, y: rbf_kernel(x, y, sigma)
    elif kernel == 'linear':
        kernel_func = linear_kernel
    else:
        raise ValueError(f"Unknown kernel type '{kernel}'")

    # Compute kernel matrices
    k_xx = kernel_func(latent1, latent1)  # Kernel between latent1 and itself
    k_yy = kernel_func(latent2, latent2)  # Kernel between latent2 and itself
    k_xy = kernel_func(latent1, latent2)  # Kernel between latent1 and latent2

    # Calculate MMD
    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd

def l_covar(latent1, latent2):
    # Concatenate the latents along the feature dimension
    combined_latents = torch.cat([latent1, latent2], dim=1)
    
    # Center the latents
    combined_latents = combined_latents - combined_latents.mean(dim=0)
    
    # Compute the covariance matrix
    cov_matrix = (combined_latents.T @ combined_latents) / (combined_latents.size(0) - 1)
    
    # Zero out the diagonal to only penalize off-diagonal elements
    cov_loss = cov_matrix.pow(2).sum() - cov_matrix.diag().pow(2).sum()
    
    return cov_loss

def l_distance_corr(latent1, latent2):
    # Pairwise distance matrices for both latents
    a = latent1.unsqueeze(0) - latent1.unsqueeze(1)
    b = latent2.unsqueeze(0) - latent2.unsqueeze(1)
    
    # Compute the norms
    a_norm = torch.norm(a, p=2, dim=-1)
    b_norm = torch.norm(b, p=2, dim=-1)
    
    # Centering the distance matrices
    A = a_norm - a_norm.mean(dim=0, keepdim=True) - a_norm.mean(dim=1, keepdim=True) + a_norm.mean()
    B = b_norm - b_norm.mean(dim=0, keepdim=True) - b_norm.mean(dim=1, keepdim=True) + b_norm.mean()
    
    # Calculate the distance correlation
    dc_loss = (A * B).mean() / (A.pow(2).mean().sqrt() * B.pow(2).mean().sqrt())
    
    return dc_loss

def l_barlow_twins(latent1, latent2):
    """
    Barlow Twins loss designed to decorrelate the two latent representations by reducing redundancy
    it enforces independence by minimizing off-diagonal elements in the cross-correlation matrix.
    """
    # Normalize latents
    latent1 = (latent1 - latent1.mean(0)) / latent1.std(0)
    latent2 = (latent2 - latent2.mean(0)) / latent2.std(0)

    # Cross-correlation matrix
    c = torch.mm(latent1.T, latent2) / latent1.size(0)
    
    # On- and off-diagonal terms
    on_diag = torch.diagonal(c).pow(2).sum()
    off_diag = (c - torch.eye(c.size(0), device=c.device)).pow(2).sum()
    
    loss = on_diag + 0.01 * off_diag
    return loss