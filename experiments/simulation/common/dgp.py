"""Traffic-light image DGP: three circles encode a confounded, a mediated, and an independent signal."""
import torch


def circle_mask(h, w, center, radius):
    """Boolean (h, w) mask of a filled circle."""
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist = (X - center[1]) ** 2 + (Y - center[0]) ** 2
    return dist <= radius ** 2


def split_mask_into_vertical_strips(mask, n_strips):
    """Split a boolean mask into n_strips vertical sub-masks of near-equal width."""
    ys, xs = mask.nonzero(as_tuple=True)
    x_min, x_max = xs.min(), xs.max() + 1
    width = x_max - x_min
    base = width // n_strips
    remainder = width % n_strips

    strips = []
    start = x_min
    for i in range(n_strips):
        end = start + base + (1 if i < remainder else 0)
        strip = torch.zeros_like(mask)
        strip[ys, xs] = (xs >= start) & (xs < end)
        strips.append(strip)
        start = end
    return strips


def simulate_traffic_light_data(
        n=800, h=20, w=60, circle_radius=8,
        bz=1., b2=1., b3=1., cv1=0.8, cv2=0.5, sdy=1., seed=0,
        n_covars=1, outcome_type='continuous'):
    """Draw (X, Z, y, fx, fz, fr): images X of three circles, covariates Z, linear fz."""
    torch.manual_seed(seed)

    # covariates and latents
    Z = torch.rand(n, n_covars)
    v1_raw = torch.rand(n, n_covars)
    v2_raw = torch.rand(n, n_covars)
    v3 = torch.rand(n, 1)
    v1 = (1 - cv1) * v1_raw + cv1 * Z    # confounded, not in y
    v2 = (1 - cv2) * v2_raw + cv2 * Z    # confounded, in y
    # v3 independent, in y

    # images
    X = torch.zeros((n, 1, h, w))
    centers = [(h // 2, w // 6), (h // 2, w // 2), (h // 2, 5 * w // 6)]
    mask1_strips = split_mask_into_vertical_strips(circle_mask(h, w, centers[0], circle_radius), n_covars)
    mask2_strips = split_mask_into_vertical_strips(circle_mask(h, w, centers[1], circle_radius), n_covars)
    mask3 = circle_mask(h, w, centers[2], circle_radius)
    for i in range(n):
        for j, strip in enumerate(mask1_strips):
            X[i, 0][strip] = v1[i, j]
        for j, strip in enumerate(mask2_strips):
            X[i, 0][strip] = v2[i, j]
        X[i, 0][mask3] = v3[i]

    # outcome
    # coefficients scale with sqrt(p) so signal variance is constant in n_covars
    b2 = b2 * n_covars ** 0.5
    bz = bz * n_covars ** 0.5
    fx = b2 * (v2 - 0.5).mean(dim=1, keepdim=True) + b3 * (v3 - 0.5)
    fz = bz * (Z - 0.5).mean(dim=1, keepdim=True)
    eta = fx + fz
    if outcome_type == 'continuous':
        y = eta + sdy * torch.randn(n, 1)
    elif outcome_type == 'binary':
        y = torch.bernoulli(torch.sigmoid(eta))
    else:
        raise ValueError("outcome_type must be 'continuous' or 'binary'.")

    # residual image effect: fx minus the part mediated through Z
    fr = fx - b2 * cv2 * (Z - 0.5).mean(dim=1, keepdim=True)

    return X, Z, y, fx, fz, fr
