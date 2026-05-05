import numpy as np
import torch
from scipy.interpolate import BSpline


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


# ─── Nonlinear-fz DGP and B-spline covariate basis ────────────────────────────
#
# A second DGP variant where Z enters y nonlinearly via a sine. The image
# half is identical to `simulate_traffic_light_data`; only fz changes.
# Used by the `nonlinear_fz` block of run_full_simulation.py, paired with
# a B-spline basis transform of Z so the post-hoc model fits a spline
# regression for fz with no model changes.


def simulate_data_nonlinear_fz(
        n=800, h=20, w=60, circle_radius=8,
        bz=1., b2=1., b3=1., cv1=0.5, cv2=0.5, sdy=1., seed=0,
        n_covars=1, outcome_type='continuous'):
    """Same image DGP as `simulate_traffic_light_data` (univariate Z),
    but with a nonlinear sinusoidal covariate effect:

        fz(Z) = bz * sin(2π * (Z - 0.5))

    one full period over Z's support [0, 1], centred about Z = 0.5.
    Only n_covars=1 is supported.
    """
    if n_covars != 1:
        raise ValueError(
            "simulate_data_nonlinear_fz requires n_covars=1 (sin fz is univariate)"
        )

    torch.manual_seed(seed)
    Z = torch.rand(n, 1)
    v1_raw = torch.rand(n, 1)
    v2_raw = torch.rand(n, 1)
    v3 = torch.rand(n, 1)
    v1 = (1 - cv1) * v1_raw + cv1 * Z
    v2 = (1 - cv2) * v2_raw + cv2 * Z

    X = torch.zeros((n, 1, h, w))
    centers = [(h // 2, w // 6), (h // 2, w // 2), (h // 2, 5 * w // 6)]
    masks = [circle_mask(h, w, c, circle_radius) for c in centers]
    for i in range(n):
        X[i, 0][masks[0]] = v1[i]
        X[i, 0][masks[1]] = v2[i]
        X[i, 0][masks[2]] = v3[i]

    fx = b2 * (v2 - 0.5) + b3 * (v3 - 0.5)
    fz = bz * torch.sin(2 * torch.pi * (Z - 0.5))   # nonlinear
    eta = fx + fz

    if outcome_type == 'continuous':
        y = eta + sdy * torch.randn(n, 1)
    elif outcome_type == 'binary':
        p = torch.sigmoid(eta)
        y = torch.bernoulli(p)
    else:
        raise ValueError("outcome_type must be 'continuous' or 'binary'.")

    # Residual fx: v2's Z-dependence is still linear (v2 = (1-c2)U + c2 Z),
    # so the same expression as the linear-fz case applies. Changing fz only
    # changes how Z enters y, not how Z enters X.
    fr = fx - b2 * cv2 * (Z - 0.5)
    return X, Z, y, fx, fz, fr


def make_bspline_basis(z, knots, degree=3):
    """Cubic B-spline design matrix for a 1-d covariate. Mirrors the
    `ToSplineDesign` construction in the proj-orthogonalisation ADNI
    cross-fit script:

      - Knot vector clamped: degree+1 repetitions of each endpoint, plus
        the inner knots from `knots[1:-1]`.
      - n_basis = len(knot_vector) - degree - 1.
      - Out-of-support points are clipped to [knots[0], knots[-1]] so
        boundary samples get a non-zero basis row (vs the silent zero of
        the unclamped extrapolate=False call).

    Parameters
    ----------
    z      : 1-d numpy array (or anything `np.atleast_1d` accepts).
    knots  : 1-d array of length K+2 — the unique knots [k_min, ..., k_max];
             inner knots = knots[1:-1].
    degree : spline degree (default 3 = cubic).

    Returns
    -------
    Ndarray of shape (n, n_basis) with dtype float32.
    """
    z = np.atleast_1d(z).astype(np.float32)
    knot_vector = np.concatenate([
        np.repeat(knots[0], degree + 1),
        knots[1:-1],
        np.repeat(knots[-1], degree + 1),
    ])
    n_basis = len(knot_vector) - degree - 1
    lo = knot_vector[degree]
    hi = knot_vector[-degree - 1]
    z_clipped = np.clip(z, lo, hi)
    B = np.column_stack([
        BSpline.basis_element(
            knot_vector[i:i + degree + 2], extrapolate=False
        )(z_clipped)
        for i in range(n_basis)
    ])
    return np.nan_to_num(B, nan=0.0).astype(np.float32)


class BSplineBasisTransform:
    """Picklable callable that maps a covariate tensor (n, 1) to its
    B-spline basis tensor (n, n_basis). Module-level so it survives the
    spawn-based multiprocessing pool (closures with state can fail to
    pickle; a class with picklable attributes always works)."""

    def __init__(self, knots, degree=3):
        self.knots = np.asarray(knots, dtype=np.float32)
        self.degree = degree
        # Match the basis count derived in make_bspline_basis so
        # consumers can size the model's num_covariates without
        # constructing the basis first.
        self.n_basis = len(self.knots) + degree - 1

    def __call__(self, Z):
        z = Z.numpy().ravel() if torch.is_tensor(Z) else np.asarray(Z).ravel()
        return torch.from_numpy(make_bspline_basis(z, self.knots, self.degree))
