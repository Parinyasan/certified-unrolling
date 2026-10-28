import numpy as np
from cu.algos.certificates import gap_certificate, run_ista_certified, select_start
rng = np.random.default_rng(1)
m, n, s = 50, 100, 8
A = rng.standard_normal((m, n)) / np.sqrt(m)
sv = np.linalg.svd(A, compute_uv=False); smax2 = sv[0]**2
eps = 1e-2 * smax2; mu = 2 / smax2
xs = np.zeros(n); idx = rng.choice(n, s, replace=False); xs[idx] = rng.standard_normal(s)
y = A @ xs + 0.01 * rng.standard_normal(m)
lam = 0.1 * np.max(np.abs(A.T @ y))
def T(x):
    v = x - mu * A.T @ (A @ x - y)
    return np.sign(v) * np.maximum(np.abs(v) - mu * lam, 0) / (1 + mu * eps)
xh = np.zeros(n)
for _ in range(20000):
    xh = T(xh)
_, _, Vt = np.linalg.svd(A); N = Vt[m:].T; R = Vt[:m].T
starts = []
for t in np.logspace(-2, 0.3, 12):
    for kind in ("null", "row", "offsupp", "rand"):
        for rep in range(3):
            if kind == "null": v = N @ rng.standard_normal(n - m)
            elif kind == "row": v = R @ rng.standard_normal(m)
            elif kind == "offsupp":
                v = np.zeros(n); off = np.where(xh == 0)[0]; v[off] = rng.standard_normal(off.size)
            else: v = rng.standard_normal(n)
            v *= t * np.linalg.norm(xh) / np.linalg.norm(v)
            starts.append(xh + v)
starts.append(np.zeros(n))
X0 = np.array(starts); B = X0.shape[0]; Y = np.tile(y, (B, 1))
G, _ = gap_certificate(A, Y, X0, np.full(B, lam), eps)
run = run_ista_certified(A, Y, np.full(B, lam), eps, mu, X0, 1e-3)
k = run.k_star; d = np.linalg.norm(X0 - xh, axis=1)
disc = [(i, j) for i in range(B) for j in range(B) if G[i] < G[j] and k[i] > k[j]]
print("B =", B, " discordant ordered pairs (G_i<G_j but k_i>k_j):", len(disc), "of", B * (B - 1) // 2, "unordered pairs")
best = max(disc, key=lambda p: k[p[0]] - k[p[1]])
i, j = best
print(f"pair: G={G[i]:.4g} (d={d[i]:.4g}) -> k*={k[i]}   vs   G={G[j]:.4g} (d={d[j]:.4g}) -> k*={k[j]}")
z = B - 1
hz = [i for i in range(B - 1) if G[i] <= G[z] and k[i] > k[z]]
print("starts kept by select_start (G(x0)<=G(0)) whose tail is LONGER than the cold tail:", len(hz), "of", sum(G[:-1] <= G[z]), "kept;  cold k* =", k[z])
if hz:
    i = max(hz, key=lambda q: k[q])
    print(f"  example: G(x0)={G[i]:.4g} <= G(0)={G[z]:.4g}, d(x0)={d[i]:.4g} vs d(0)={d[z]:.4g}, k*(x0)={k[i]} > k*(0)={k[z]}")
from scipy.stats import spearmanr
print("Spearman(G, k*) =", spearmanr(G, k).correlation, "; Spearman(d, k*) =", spearmanr(d, k).correlation)
kinds = ("null", "row", "offsupp", "rand"); ts = np.logspace(-2, 0.3, 12)
for q in hz:
    ti, rem = divmod(q, 12); ki, rep = divmod(rem, 3)
    print(f"  kept-but-slower start #{q}: kind={kinds[ki]}, ||delta||/||xhat||={ts[ti]:.3f}, G={G[q]:.4g}, k*={k[q]}")
xsel, keep, Gh, G0 = select_start(A, Y[:-1], X0[:-1], np.full(B-1, lam), eps)
print("select_start agrees with G<=G(0):", np.array_equal(keep, G[:-1] <= G[z]))
