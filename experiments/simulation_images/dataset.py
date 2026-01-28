import torch

def simulate_traffic_light_data(
        n=800, h=20, w=60, circle_radius=8,
        bz=1., b2=1., b3=1., cv1=0.8, cv2=0.5, sdy=1., seed=0,
        n_covars=1, outcome_type='continuous'):

    torch.manual_seed(seed)

    # -------------------------------------------------
    # 1. Covariates Z (Uniform[0,1])
    # -------------------------------------------------
    Z = torch.rand(n, n_covars)

    # -------------------------------------------------
    # 2. Latent variables
    # -------------------------------------------------
    v1_raw = torch.rand(n, n_covars)
    v2_raw = torch.rand(n, n_covars)
    v3 = torch.rand(n, 1)  # independent

    # Correlate v1 and v2 with Z
    v1 = torch.zeros((n, n_covars))
    v2 = torch.zeros((n, n_covars))
    for j in range(n_covars):
        v1[:, j] = (1 - cv1) * v1_raw[:, j] + cv1 * Z[:, j]
        v2[:, j] = (1 - cv2) * v2_raw[:, j] + cv2 * Z[:, j]

    # -------------------------------------------------
    # 3. Build X images
    # -------------------------------------------------
    X = torch.zeros((n, 1, h, w))

    centers = [(h // 2, w // 6), (h // 2, w // 2), (h // 2, 5 * w // 6)]

    mask1 = circle_mask(h, w, centers[0], circle_radius)
    mask2 = circle_mask(h, w, centers[1], circle_radius)
    mask3 = circle_mask(h, w, centers[2], circle_radius)

    # For multiple confounders: Split confounded masks into vertical strips.
    mask1_strips = split_mask_into_vertical_strips(mask1, n_covars)
    mask2_strips = split_mask_into_vertical_strips(mask2, n_covars)
    
    # Fill images
    for i in range(n):
        # v1 circle: confounded by Zs, split into strips
        for j, jstrip in enumerate(mask1_strips):
            X[i, 0][jstrip] = v1[i, j]
        # v2 circle: confounded by Zs, split into strips
        for j, jstrip in enumerate(mask2_strips):
            X[i, 0][jstrip] = v2[i, j]
        # v3 circle: unconfounded
        X[i, 0][mask3] = v3[i]

    # -------------------------------------------------
    # 4. Outcome generation (NOT CENTERED)
    # -------------------------------------------------
    # Adjust coefficients for number of covariates
    b2 = b2 * n_covars**0.5
    bz = bz * n_covars**0.5

    # Base linear predictor.
    fx = b2 * (v2 - 0.5).mean(dim=1, keepdim=True) + b3 * (v3 - 0.5)
    fz = bz * (Z - 0.5).mean(dim=1, keepdim=True)
    eta = fx + fz

    # Response generation.
    if outcome_type == 'continuous':
        y = eta + sdy * torch.randn(n, 1)
    elif outcome_type == 'binary':
        p = torch.sigmoid(eta)
        y = torch.bernoulli(p)
    else:
        raise ValueError("outcome_type must be 'continuous' or 'binary'.")

    # -------------------------------------------------
    # 5. Residual (unconfounded) effect (CENTERED)
    # -------------------------------------------------
    fr = fx - b2 * cv2 * (Z - 0.5).mean(dim=1, keepdim=True)

    return X, Z, y, fx, fz, fr


def circle_mask(h, w, center, radius):
    Y, Xg = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing='ij'
    )
    dist = (Xg - center[1])**2 + (Y - center[0])**2
    return dist <= radius**2


def split_mask_into_vertical_strips(mask, n_strips):
    """
    Splits a boolean mask into n_strips vertical sub-masks.
    Returns a list of boolean masks.
    """
    ys, xs = mask.nonzero(as_tuple=True)
    x_min, x_max = xs.min(), xs.max() + 1
    width = x_max - x_min

    base = width // n_strips
    remainder = width % n_strips

    strips = []
    start = x_min
    for i in range(n_strips):
        w_i = base + (1 if i < remainder else 0)
        end = start + w_i

        strip = torch.zeros_like(mask)
        strip[ys, xs] = (xs >= start) & (xs < end)
        strips.append(strip)

        start = end

    return strips
