import torch

def simulate_traffic_light_data(n=800, h=20, w=60, circle_radius=8,
                                bz=1., b2=1., b3=1., cv1=0.8, cv2=0.5, sdy=1., seed=0):

    torch.manual_seed(seed)

    # 1. Covariate Z
    Z = torch.rand(n, 1)  # uniform(0,1)

    # 2. Latent variables
    v1_raw = torch.rand(n, 1)
    v2_raw = torch.rand(n, 1)
    v3 = torch.rand(n, 1) # independent

    # Correlate v1 and v2 with Z
    v1 = (1 - cv1) * v1_raw + cv1 * Z
    v2 = (1 - cv2) * v2_raw + cv2 * Z

    # 3. Build X images.
    X = torch.zeros((n, 1, h, w))

    # Circle centers (left, center, right)
    centers = [(h//2, w//6), (h//2, w//2), (h//2, 5*w//6)]

    # Create circle mask function
    def circle_mask(h, w, center, radius):
        Y, Xg = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist = (Xg - center[1])**2 + (Y - center[0])**2
        return dist <= radius**2

    mask1 = circle_mask(h, w, centers[0], circle_radius)
    mask2 = circle_mask(h, w, centers[1], circle_radius)
    mask3 = circle_mask(h, w, centers[2], circle_radius)

    # Paint circles per sample in grayscale according to latent values
    for i in range(n):
        X[i, 0][mask1] = v1[i]
        X[i, 0][mask2] = v2[i]
        X[i, 0][mask3] = v3[i]

    # 4. Outcome y depends on v2, v3, and Z with balanced magnitudes
    y = b2 * v2 + b3 * v3 + bz * Z + sdy*torch.randn(n, 1)

    # 5. Also save effects (centred):
    fx = b2 * (v2 - 0.5) + b3 * (v3 - 0.5)
    fz = bz * (Z - 0.5)
    fr = b2 * (v2 - 0.5 - cv2*(Z - 0.5)) + b3 * (v3 - 0.5)

    return X, Z, y, fx, fz, fr
