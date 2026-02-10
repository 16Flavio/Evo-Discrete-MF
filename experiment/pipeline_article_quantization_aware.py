import os, io, json, zlib, math, time, sys
# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.solver import metaheuristic
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# 0) Bitstream: reproduction utils.py (dict_to_bytes/combine_bytes/encode_tensor)
# ============================================================
def dict_to_bytes(d):
    return json.dumps(d).encode("utf-8")

def _combine_two(p1: bytes, p2: bytes) -> bytes:
    if len(p1) > 0xFFFFFFFF:
        raise ValueError("payload too large")
    return len(p1).to_bytes(4, byteorder="big") + p1 + p2

def combine_bytes(payloads):
    out = payloads[0]
    for p in payloads[1:]:
        out = _combine_two(out, p)
    return out

def encode_matrix_numpy(matrix_2d: np.ndarray, mode="col") -> bytes:
    A = np.asarray(matrix_2d)
    if A.ndim != 2:
        raise ValueError("encode_matrix_numpy expects 2D array")
    if mode not in ("col", "row"):
        raise ValueError("mode must be 'col' or 'row'")

    if mode == "col":
        fibers = [A[:, k:k+1] for k in range(A.shape[1])]  # (M,1)
    else:
        fibers = [A[k:k+1, :] for k in range(A.shape[0])]

    encoded_fibers = []
    for f in fibers:
        b = np.ascontiguousarray(f).tobytes()
        encoded_fibers.append(zlib.compress(b, level=9))

    metadata = {
        "num_fibers": len(fibers),
        "mode": mode,
        "dtype": str(A.dtype),
    }
    encoded_metadata = dict_to_bytes(metadata)
    encoded_fibers_blob = combine_bytes(encoded_fibers)
    encoded_matrix = combine_bytes([encoded_metadata, encoded_fibers_blob])
    return encoded_matrix

def encode_tensor_numpy(arr: np.ndarray) -> bytes:
    A = np.asarray(arr)
    if A.ndim == 2:
        return encode_matrix_numpy(A, mode="col")

    raw = np.ascontiguousarray(A).tobytes()
    comp = zlib.compress(raw, level=9)
    metadata = {"shape": list(A.shape), "dtype": str(A.dtype)}
    encoded_metadata = dict_to_bytes(metadata)
    return combine_bytes([encoded_metadata, comp])


# ============================================================
# 1) Métriques (PSNR) + bpp
# ============================================================
def psnr_uint8_rgb(x_hat_uint8, x_uint8):
    xh = x_hat_uint8.astype(np.float64)
    x  = x_uint8.astype(np.float64)
    mse = np.mean((xh - x) ** 2)
    if mse < 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))

def compress_one_image_jpeg_default(rgb_uint8, quality):
    """JPEG Pillow 'par défaut' (le plus fidèle à 'we used Pillow' sans tuning)."""
    H, W, _ = rgb_uint8.shape
    img = Image.fromarray(rgb_uint8, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))  # <-- pas d'options
    bitstream = buf.getvalue()
    rec = Image.open(io.BytesIO(bitstream)).convert("RGB")
    rgb_hat = np.array(rec, dtype=np.uint8)
    bpp = (8.0 * len(bitstream)) / (H * W)
    return bpp, float(psnr_uint8_rgb(rgb_hat, rgb_uint8))

def compress_one_image_jpeg_optimized(rgb_uint8, quality):
    """JPEG Pillow avec options qui rendent souvent JPEG 'trop bon' à bas bpp."""
    H, W, _ = rgb_uint8.shape
    img = Image.fromarray(rgb_uint8, mode="RGB")
    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=int(quality),
        optimize=True,
        progressive=False,
        subsampling=2,   # 4:2:0
    )
    bitstream = buf.getvalue()
    rec = Image.open(io.BytesIO(bitstream)).convert("RGB")
    rgb_hat = np.array(rec, dtype=np.uint8)
    bpp = (8.0 * len(bitstream)) / (H * W)
    return bpp, float(psnr_uint8_rgb(rgb_hat, rgb_uint8))

def bpp_from_bytes(byte_stream: bytes, H: int, W: int) -> float:
    return (8.0 * len(byte_stream)) / (H * W)


# ============================================================
# 2) RGB <-> YCbCr (identique aux formules utils.py)
# ============================================================
def rgb_to_ycbcr(rgb_uint8):
    rgb = rgb_uint8.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y  =  0.299    * R + 0.587    * G + 0.114    * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5      * B + 128.0
    Cr =  0.5      * R - 0.418688 * G - 0.081312 * B + 128.0
    return Y, Cb, Cr

def ycbcr_to_rgb(Y, Cb, Cr):
    Y  = Y.astype(np.float64)
    Cb = Cb.astype(np.float64) - 128.0
    Cr = Cr.astype(np.float64) - 128.0
    R = Y + 1.40200 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.77200 * Cb
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ============================================================
# 3) Chroma down/up: "area" pour scale_factor 0.5 et tailles paires == avg 2x2
# ============================================================
def downsample_2x2_area(X):
    H, W = X.shape
    if (H % 2) != 0 or (W % 2) != 0:
        H2 = (H // 2) * 2
        W2 = (W // 2) * 2
        X = X[:H2, :W2]
        H, W = X.shape
    return 0.25 * (X[0::2, 0::2] + X[1::2, 0::2] + X[0::2, 1::2] + X[1::2, 1::2])

def upsample_nearest_2x(X, out_hw):
    Y = np.repeat(np.repeat(X, 2, axis=0), 2, axis=1)
    H, W = out_hw
    return Y[:H, :W]


# ============================================================
# 4) Pad/unpad CENTRÉ reflect comme utils.py
# ============================================================
defP = 8
def pad_image_center_reflect(X, patch_size=(8, 8)):
    H, W = X.shape
    P, Q = patch_size
    pad_h = (P - H % P) % P
    pad_w = (Q - W % Q) % Q
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    Xp = np.pad(X, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
    return Xp, (pad_top, pad_left), (H, W), Xp.shape

def unpad_image_center(Xp, pad_top_left, orig_hw):
    pad_top, pad_left = pad_top_left
    H0, W0 = orig_hw
    return Xp[pad_top:pad_top + H0, pad_left:pad_left + W0]

def patchify_8x8_centered(X):
    Xp, pad_tl, orig_hw, padded_hw = pad_image_center_reflect(X, (8, 8))
    Hp, Wp = padded_hw
    patches = []
    for i in range(0, Hp, 8):
        for j in range(0, Wp, 8):
            patches.append(Xp[i:i+8, j:j+8].reshape(-1))
    M = np.stack(patches, axis=0)  # (num_patches, 64)
    return M, pad_tl, orig_hw, padded_hw

def depatchify_8x8_centered(M, padded_hw):
    Hp, Wp = padded_hw
    Xp = np.zeros((Hp, Wp), dtype=np.float64)
    idx = 0
    for i in range(0, Hp, 8):
        for j in range(0, Wp, 8):
            Xp[i:i+8, j:j+8] = M[idx].reshape(8, 8)
            idx += 1
    return Xp


# ============================================================
# 5) IMF (QMF sans w) fidèle "factor=(0,1)" + bounds clamp
# ============================================================
def method_IMF_from_quantization_aware(X, r, bounds=(-16, 15), num_iters=10):
    X = np.asarray(X, dtype=np.float64)
    M, N = X.shape
    r = int(r)
    R = min(r, M, N)
    lo = int(math.ceil(bounds[0]))
    hi = int(math.floor(bounds[1]))
    eps = 1e-12

    U0, s, Vt0 = np.linalg.svd(X, full_matrices=False)
    U0 = U0[:, :R]
    s  = s[:R]
    Vt0 = Vt0[:R, :]
    s_sqrt = np.sqrt(s)
    U = U0 * s_sqrt[None, :]
    V = (Vt0.T) * s_sqrt[None, :]

    if r > R:
        U = np.pad(U, ((0, 0), (0, r - R)), mode="constant")
        V = np.pad(V, ((0, 0), (0, r - R)), mode="constant")

    for _ in range(num_iters):
        A = X @ V
        B = V.T @ V
        for rr in range(r):
            denom = B[rr, rr]
            if abs(denom) < eps:
                continue
            if r == 1:
                term2 = 0.0
            else:
                idx = [j for j in range(r) if j != rr]
                term2 = (U[:, idx] @ B[np.ix_(idx, [rr])])[:, 0]
            u_new = (A[:, rr] - term2) / denom
            U[:, rr] = np.clip(np.round(u_new), lo, hi)

        A = X.T @ U
        B = U.T @ U
        for rr in range(r):
            denom = B[rr, rr]
            if abs(denom) < eps:
                continue
            if r == 1:
                term2 = 0.0
            else:
                idx = [j for j in range(r) if j != rr]
                term2 = (V[:, idx] @ B[np.ix_(idx, [rr])])[:, 0]
            v_new = (A[:, rr] - term2) / denom
            V[:, rr] = np.clip(np.round(v_new), lo, hi)

    return U.astype(np.int8, copy=False), V.astype(np.int8, copy=False)


# ============================================================
# 5bis) Algorithme Evolutionnaire
# ============================================================
def my_method(X, r, bounds=(-16, 15), tmax=2.0):
    """
    X: np.ndarray float64, shape (M,N)
    r: int
    bounds: (lo,hi) clamp
    tmax: budget temps en secondes

    Doit retourner:
        U_int : (M,r) entier
        V_int : (N,r) entier
    """
    LW = int(math.ceil(bounds[0]))
    UW = int(math.floor(bounds[1]))
    LH = LW
    UH = UW
    
    # Calling the evolutionary solver
    W, H, f = metaheuristic(X, r, LW, UW, LH, UH, 'IMF', TIME_LIMIT=tmax, N=20, debug_mode=False)
    
    # Return U and V such that X ~ U @ V.T
    # W is (M, r), H is (r, N) => U=W, V=H.T
    return W.astype(np.int8, copy=False), H.T.astype(np.int8, copy=False)

# ============================================================
# 6) SVD baseline: quantize/dequantize identiques utils.py
# ============================================================
def quantize_like_repo(x_float64, target_dtype=np.int8):
    x = np.asarray(x_float64, dtype=np.float64)

    if target_dtype == np.int8:
        qmin, qmax = -128, 127
    elif target_dtype == np.uint8:
        qmin, qmax = 0, 255
    else:
        raise ValueError("only int8/uint8 here")

    min_val = float(np.min(x))
    max_val = float(np.max(x))
    if abs(max_val - min_val) < 1e-12:
        scale = 1.0
        q = np.full_like(x, qmin, dtype=target_dtype)
        return q, np.float32(scale), np.float32(min_val)

    scale = (max_val - min_val) / float(qmax - qmin)
    q = (x - min_val) / scale + qmin
    q = np.clip(q, qmin, qmax).astype(target_dtype)
    return q, np.float32(scale), np.float32(min_val)

def dequantize_like_repo(q_int, scale, min_val):
    q = np.asarray(q_int).astype(np.float32)
    return (q - float(np.min(q))) * float(scale) + float(min_val)

def method_SVD_from_quantization_aware(X, r):
    X = np.asarray(X, dtype=np.float64)
    M, N = X.shape
    r = int(r)
    R = min(r, M, N)

    U0, s, Vt0 = np.linalg.svd(X, full_matrices=False)
    U0 = U0[:, :R]
    s  = s[:R]
    Vt0 = Vt0[:R, :]
    s_sqrt = np.sqrt(s)
    U = U0 * s_sqrt[None, :]
    V = (Vt0.T) * s_sqrt[None, :]

    if r > R:
        U = np.pad(U, ((0, 0), (0, r - R)), mode="constant")
        V = np.pad(V, ((0, 0), (0, r - R)), mode="constant")

    Uq, su, minu = quantize_like_repo(U, np.int8)
    Vq, sv, minv = quantize_like_repo(V, np.int8)

    return Uq, Vq, su, minu, sv, minv


# ============================================================
# 7) Codec "comme qmf_encode" (bytes) + reconstruction
# ============================================================
def compress_one_image_with_method_like_repo(rgb_uint8, r, method_name, tmax_my=2.0):
    H, W, _ = rgb_uint8.shape

    Y, Cb, Cr = rgb_to_ycbcr(rgb_uint8)
    Cb_ds = downsample_2x2_area(Cb)
    Cr_ds = downsample_2x2_area(Cr)

    XY,  padY,  origY,  paddedY  = patchify_8x8_centered(Y)
    XCb, padCb, origCb, paddedCb = patchify_8x8_centered(Cb_ds)
    XCr, padCr, origCr, paddedCr = patchify_8x8_centered(Cr_ds)

    ry = int(r)
    rc = max(int(r) // 2, 1)

    metadata = {
        "dtype": "uint8",
        "color space": "YCbCr",
        "patch": True,
        "bounds": (-16, 15),
        "patch size": (8, 8),
        "original size": [list(origY), list(origCb), list(origCr)],
        "padded size": [list(paddedY), list(paddedCb), list(paddedCr)],
        "rank": [ry, rc, rc],
    }
    encoded_metadata = dict_to_bytes(metadata)

    # IMF
    if method_name == "IMF":
        Uy, Vy = method_IMF_from_quantization_aware(XY,  ry, bounds=(-16, 15), num_iters=10)
        Uc, Vc = method_IMF_from_quantization_aware(XCb, rc, bounds=(-16, 15), num_iters=10)
        Ur, Vr = method_IMF_from_quantization_aware(XCr, rc, bounds=(-16, 15), num_iters=10)

        factors = [Uy, Vy, Uc, Vc, Ur, Vr]

        XY_hat  = Uy.astype(np.float64) @ Vy.astype(np.float64).T
        XCb_hat = Uc.astype(np.float64) @ Vc.astype(np.float64).T
        XCr_hat = Ur.astype(np.float64) @ Vr.astype(np.float64).T

    # SVD baseline
    elif method_name == "SVD":
        Uyq, Vyq, su_y, minu_y, sv_y, minv_y = method_SVD_from_quantization_aware(XY,  ry)
        Ucq, Vcq, su_c, minu_c, sv_c, minv_c = method_SVD_from_quantization_aware(XCb, rc)
        Urq, Vrq, su_r, minu_r, sv_r, minv_r = method_SVD_from_quantization_aware(XCr, rc)

        factors = [Uyq, Vyq, Ucq, Vcq, Urq, Vrq]

        Uy = dequantize_like_repo(Uyq, su_y, minu_y)
        Vy = dequantize_like_repo(Vyq, sv_y, minv_y)
        Uc = dequantize_like_repo(Ucq, su_c, minu_c)
        Vc = dequantize_like_repo(Vcq, sv_c, minv_c)
        Ur = dequantize_like_repo(Urq, su_r, minu_r)
        Vr = dequantize_like_repo(Vrq, sv_r, minv_r)

        XY_hat  = Uy @ Vy.T
        XCb_hat = Uc @ Vc.T
        XCr_hat = Ur @ Vr.T

        qparams = [
            np.array([su_y, minu_y], dtype=np.float32),
            np.array([sv_y, minv_y], dtype=np.float32),
            np.array([su_c, minu_c], dtype=np.float32),
            np.array([sv_c, minv_c], dtype=np.float32),
            np.array([su_r, minu_r], dtype=np.float32),
            np.array([sv_r, minv_r], dtype=np.float32),
        ]

    # TA méthode (même interface que IMF : facteurs entiers bornés)
    elif method_name == "EvoMF":
        Uy, Vy = my_method(XY,  ry, bounds=(-16, 15), tmax=tmax_my)
        Uc, Vc = my_method(XCb, rc, bounds=(-16, 15), tmax=tmax_my)
        Ur, Vr = my_method(XCr, rc, bounds=(-16, 15), tmax=tmax_my)

        # on s'assure juste que c'est bien int8 (borné) pour encoder pareil que IMF
        Uy = np.asarray(Uy, dtype=np.int8)
        Vy = np.asarray(Vy, dtype=np.int8)
        Uc = np.asarray(Uc, dtype=np.int8)
        Vc = np.asarray(Vc, dtype=np.int8)
        Ur = np.asarray(Ur, dtype=np.int8)
        Vr = np.asarray(Vr, dtype=np.int8)

        factors = [Uy, Vy, Uc, Vc, Ur, Vr]

        XY_hat  = Uy.astype(np.float64) @ Vy.astype(np.float64).T
        XCb_hat = Uc.astype(np.float64) @ Vc.astype(np.float64).T
        XCr_hat = Ur.astype(np.float64) @ Vr.astype(np.float64).T

    else:
        raise ValueError("method_name must be 'IMF' or 'SVD' or 'MY'")

    encoded_factors_list = [encode_tensor_numpy(f) for f in factors]

    if method_name == "SVD":
        encoded_factors_list += [encode_tensor_numpy(qp) for qp in qparams]

    encoded_factors = combine_bytes(encoded_factors_list)
    encoded_image = combine_bytes([encoded_metadata, encoded_factors])

    Yp_hat     = depatchify_8x8_centered(XY_hat,  paddedY)
    Cbp_hat_ds = depatchify_8x8_centered(XCb_hat, paddedCb)
    Crp_hat_ds = depatchify_8x8_centered(XCr_hat, paddedCr)

    Y_hat     = unpad_image_center(Yp_hat, padY, origY)
    Cb_hat_ds = unpad_image_center(Cbp_hat_ds, padCb, origCb)
    Cr_hat_ds = unpad_image_center(Crp_hat_ds, padCr, origCr)

    Cb_hat = upsample_nearest_2x(Cb_hat_ds, (H, W))
    Cr_hat = upsample_nearest_2x(Cr_hat_ds, (H, W))

    rgb_hat = ycbcr_to_rgb(Y_hat, Cb_hat, Cr_hat)

    bpp = bpp_from_bytes(encoded_image, H, W)
    p = psnr_uint8_rgb(rgb_hat, rgb_uint8)
    return bpp, p


# ============================================================
# 9) LOESS (comme leur Plot/LOESS: degree=1, frac grid, LOOCV)
# ============================================================
def loess_predict_with_frac(x, y, x_new, frac):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64)
    n = len(x)
    k = int(np.ceil(frac * n))
    k = max(2, min(k, n))

    y_new = np.zeros_like(x_new, dtype=np.float64)

    for i, xn in enumerate(x_new):
        d = np.abs(x - xn)
        idx = np.argsort(d)[:k]
        xk = x[idx]
        yk = y[idx]
        dk = d[idx]
        dmax = dk[-1] if dk[-1] > 0 else 1.0
        u = dk / dmax
        w = np.clip((1 - u**3) ** 3, 0, 1)

        Xmat = np.vstack([xk, np.ones_like(xk)]).T
        Wmat = np.diag(w)
        beta = np.linalg.lstsq(Wmat @ Xmat, Wmat @ yk, rcond=None)[0]
        a, b = beta[0], beta[1]
        y_new[i] = a * xn + b

    return y_new

def loess_best_predict(x, y, x_new):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return np.full_like(x_new, np.nan, dtype=np.float64)

    fracs = np.arange(0.15, 0.75, 0.1)
    best_frac = fracs[0]
    best_score = float("inf")

    for frac in fracs:
        errs = []
        for i in range(len(x)):
            mask = np.ones(len(x), dtype=bool)
            mask[i] = False
            yp = loess_predict_with_frac(x[mask], y[mask], np.array([x[i]]), frac)[0]
            errs.append((y[i] - yp) ** 2)
        score = float(np.mean(errs))
        if score < best_score:
            best_score = score
            best_frac = frac

    return loess_predict_with_frac(x, y, x_new, best_frac)

def standard_error(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) <= 1:
        return np.nan
    return np.std(v, ddof=1) / math.sqrt(len(v))


# ============================================================
# 10) Figure 3(a)
# ============================================================
def load_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    imgs = []
    for p in files:
        imgs.append(np.array(Image.open(p).convert("RGB"), dtype=np.uint8))
    return imgs



if __name__ == "__main__":
    kodak_folder = "kodak"
    # Ensure we look for kodak in the same directory as the script if not found in CWD
    if not os.path.exists(kodak_folder):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(script_dir, "kodak")
        if os.path.exists(cand):
            kodak_folder = cand

    images = load_images(kodak_folder)
    if len(images) == 0:
        raise RuntimeError(f"Aucune image trouvée dans {kodak_folder}.")

    bpp_grid = np.arange(0.05, 0.501, 0.05)

    jpeg_qualities = list(range(1, 96))
    ranks = list(range(1, 5)) + list(range(5, 65, 5))

    tmax_my = 30.0  # <-- budget temps (secondes) pour ta méthode

    methods = ["JPEG", "SVD", "IMF", "EvoMF"]
    per_image_points = {m: [] for m in methods}

    nbim = 0
    for rgb in images:
        nbim += 1
        print(f"{nbim}/{len(images)}")

        # JPEG
        pts = []
        for q in jpeg_qualities:
            bpp, p = compress_one_image_jpeg_default(rgb, q)
            pts.append((bpp, p))
        per_image_points["JPEG"].append(sorted(pts, key=lambda t: t[0]))

        # IMF / SVD / MY
        for m in ["IMF", "SVD", "EvoMF"]:
            pts = []
            for r in ranks:
                bpp, p = compress_one_image_with_method_like_repo(rgb, r, m, tmax_my=tmax_my)
                pts.append((bpp, p))
            per_image_points[m].append(sorted(pts, key=lambda t: t[0]))

    interp = {m: np.full((len(images), len(bpp_grid)), np.nan, dtype=np.float64) for m in methods}
    extrap = {m: np.zeros((len(images), len(bpp_grid)), dtype=bool) for m in methods}

    for i in range(len(images)):
        for m in methods:
            pts = per_image_points[m][i]
            x = np.array([t[0] for t in pts], dtype=np.float64)
            y = np.array([t[1] for t in pts], dtype=np.float64)

            order = np.argsort(x)
            x = x[order]
            y = y[order]
            _, idx_unique = np.unique(x, return_index=True)
            x = x[idx_unique]
            y = y[idx_unique]

            yhat = loess_best_predict(x, y, bpp_grid)
            interp[m][i, :] = yhat

            x_min, x_max = float(np.min(x)), float(np.max(x))
            extrap[m][i, :] = (bpp_grid < x_min) | (bpp_grid > x_max)

    mean = {}
    se = {}
    dashed = {}
    for m in methods:
        mean[m] = np.nanmean(interp[m], axis=0)
        se[m] = np.array([standard_error(interp[m][:, j]) for j in range(len(bpp_grid))])
        dashed[m] = np.all(extrap[m], axis=0)

    plt.figure()

    for m in methods:
        y = mean[m]
        s = se[m]
        d = dashed[m]

        solid = (~d) & (~np.isnan(y))
        dash  = ( d) & (~np.isnan(y))

        color = None

        if np.any(solid):
            line_solid, = plt.plot(
                bpp_grid[solid], y[solid],
                marker="o", linewidth=1, linestyle="-",
                label=m
            )
            color = line_solid.get_color()
            plt.fill_between(bpp_grid[solid], (y - s)[solid], (y + s)[solid], alpha=0.2, color=color)

        if color is None and np.any(dash):
            line_dash, = plt.plot(
                bpp_grid[dash], y[dash],
                marker="o", linewidth=1, linestyle="--",
                label=m
            )
            color = line_dash.get_color()
        elif np.any(dash):
            plt.plot(
                bpp_grid[dash], y[dash],
                marker="o", linewidth=1, linestyle="--",
                color=color, label=None
            )

        if np.any(solid) and np.any(dash):
            solid_idx = np.where(solid)[0]
            dash_idx  = np.where(dash)[0]
            i_s_last = solid_idx[-1]
            i_d_first = dash_idx[0]
            i_s_first = solid_idx[0]
            i_d_last = dash_idx[-1]

            if i_d_last < i_s_first:
                plt.plot(
                    [bpp_grid[i_d_last], bpp_grid[i_s_first]],
                    [y[i_d_last],       y[i_s_first]],
                    linestyle="--", linewidth=1, color=color
                )
            if i_s_last < i_d_first:
                plt.plot(
                    [bpp_grid[i_s_last], bpp_grid[i_d_first]],
                    [y[i_s_last],        y[i_d_first]],
                    linestyle="--", linewidth=1, color=color
                )

    plt.grid(True)
    plt.xlim(0.05, 0.50)
    plt.ylim(16, 31)
    plt.xlabel("bit rate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title("Figure 3(a) — Kodak PSNR vs bit rate")
    plt.legend(loc="lower right")
    plt.show()