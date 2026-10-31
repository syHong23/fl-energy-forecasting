"""
Differentially private synthetic load data, and an attack on the released data.

A small variational autoencoder (VAE) learns to generate pairs of consecutive days of
hourly load (48 values: the previous day and the next day), with and without DP-SGD.
The synthetic pairs are then "released" and evaluated on two axes:

  Privacy  - a distance-to-closest-record (DCR) membership inference attack: real days
             that were used for training (members) tend to lie closer to the synthetic
             data than real days from the same period that were not (non-members).
             Reported as AUC (0.5 = chance, 1.0 = perfect).
  Utility  - train on synthetic, test on real (TSTR): a ridge regression that predicts
             the next day's change from the previous day is fitted on the synthetic pairs and
             evaluated on real test pairs; plus a fidelity score (mean 1-D Wasserstein
             distance between real and synthetic values, hour by hour).

Settings:
  real_noise  - release the real member pairs with Gaussian noise (naive anonymisation)
  vae         - VAE without DP
  vae_dp      - VAE trained with DP-SGD (Opacus), one run per epsilon
  (reference) - the same regression trained on the real member pairs

Each client's data are normalised with its own statistics, as in fl_energy.py.
Usage:
  python synth.py            # UCI data
  python synth.py --synthetic
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, TensorDataset

import fl_energy as F

D = 48  # previous day + next day
LATENT = 8


def build_pairs(clients):
    """(previous day, next day) pairs in each client's normalised units."""
    tr = torch.cat([torch.cat([c["Xtr"][:, (F.HIST_DAYS - 1) * 24 : F.HIST_DAYS * 24], c["ytr"]], 1) for c in clients])
    te = torch.cat([torch.cat([c["Xte"][:, (F.HIST_DAYS - 1) * 24 : F.HIST_DAYS * 24], c["yte"]], 1) for c in clients])
    return tr, te


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2 * LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, D))
        # Gaussian decoder with a learned per-dimension noise scale, stored as a layer so DP-SGD covers it
        self.log_sigma = nn.Linear(1, D)
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.constant_(self.log_sigma.bias, -0.5)

    def sigma_log(self, n):
        return self.log_sigma(torch.ones(n, 1))

    def forward(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar

    def sample(self, n):
        with torch.no_grad():
            mean = self.dec(torch.randn(n, LATENT))
            return mean + torch.exp(self.sigma_log(n)) * torch.randn(n, D)


def vae_loss(model, x, beta):
    rec, mu, logvar = model(x)
    log_s = (model._module if hasattr(model, "_module") else model).sigma_log(len(x))
    recon = (0.5 * ((rec - x) / torch.exp(log_s)) ** 2 + log_s).sum(1)  # Gaussian negative log-likelihood
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    return (recon + beta * kl).mean()


def train_vae(members, args, epsilon=None):
    torch.manual_seed(args.seed)
    model = VAE()
    dl = DataLoader(TensorDataset(members), batch_size=args.batch, shuffle=True)
    if epsilon is None:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        from opacus import PrivacyEngine
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        pe = PrivacyEngine(accountant="rdp")
        model, opt, dl = pe.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=dl, target_epsilon=epsilon,
            target_delta=1.0 / len(members), epochs=args.epochs, max_grad_norm=args.clip)
    model.train()
    for _ in range(args.epochs):
        for (xb,) in dl:
            if len(xb) == 0:
                continue
            opt.zero_grad()
            vae_loss(model, xb, args.beta).backward()
            opt.step()
    net = model._module if epsilon is not None else model
    net.eval()
    return net


def ridge_fit(pairs, lam=10.0):
    """Ridge regression of the change from the previous day (next - previous), so that
    shrinking all weights to zero recovers the naive 'next day = previous day' forecast."""
    X = torch.cat([pairs[:, :24], torch.ones(len(pairs), 1)], 1).double()
    Y = (pairs[:, 24:] - pairs[:, :24]).double()
    W = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], dtype=torch.float64), X.T @ Y)
    return W


def ridge_mae(W, pairs):
    X = torch.cat([pairs[:, :24], torch.ones(len(pairs), 1)], 1).double()
    pred = pairs[:, :24].double() + X @ W
    return float((pred - pairs[:, 24:].double()).abs().mean())


def fidelity(real, syn):
    r, s = real.numpy(), syn.numpy()
    return float(np.mean([wasserstein_distance(r[:, j], s[:, j]) for j in range(D)]))


def dcr(points, syn, chunk=2048):
    out = []
    for i in range(0, len(points), chunk):
        out.append(torch.cdist(points[i : i + chunk], syn).min(1).values)
    return torch.cat(out).numpy()


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    ranks = pd.Series(s).rank().values
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="LD2011_2014.txt")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.5, help="weight of the KL term")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.3, help="std of the noise in the real_noise baseline")
    ap.add_argument("--epsilons", default="1,4,8")
    ap.add_argument("--attack-size", type=int, default=2000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0, help="seed for choosing clients")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.synthetic:
        df, source = F.make_synthetic(args.clients, args.seed), "synthetic"
    else:
        df, source = F.load_uci(args.data, args.clients, args.seed), "UCI ElectricityLoadDiagrams20112014"
    clients = [F.client_arrays(df[col], args.test_frac) for col in df.columns]
    train_pairs, test_pairs = build_pairs(clients)
    print(f"Data: {source} | {len(train_pairs)} training-period pairs, {len(test_pairs)} test pairs")

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        args.seed = seed
        g = torch.Generator().manual_seed(seed + 99)
        perm = torch.randperm(len(train_pairs), generator=g)
        half = len(perm) // 2
        members, non = train_pairs[perm[:half]], train_pairs[perm[half:]]
        k = min(args.attack_size, len(non))
        a_mem = members[torch.randperm(len(members), generator=g)[:k]]
        a_non = non[torch.randperm(len(non), generator=g)[:k]]
        print(f"\n=== seed {seed} | {len(members)} members, {len(non)} non-members ===")

        ref_mae = ridge_mae(ridge_fit(members), test_pairs)
        naive_mae = float((test_pairs[:, :24] - test_pairs[:, 24:]).abs().mean())

        def record(name, syn, eps=None):
            a = auc(-dcr(a_mem, syn), -dcr(a_non, syn))
            row = dict(seed=seed, setting=name, epsilon=eps, attack_auc=a,
                       tstr_mae=ridge_mae(ridge_fit(syn), test_pairs),
                       fidelity=fidelity(test_pairs, syn), trtr_mae=ref_mae, naive_mae=naive_mae)
            rows.append(row)
            print(f"  {name:<10} eps={str(eps):<4} attack AUC={a:.3f} | TSTR MAE={row['tstr_mae']:.3f} "
                  f"(real-trained {ref_mae:.3f}, naive {naive_mae:.3f}) | fidelity W1={row['fidelity']:.3f}")

        record("real_noise", members + args.noise * torch.randn(members.shape, generator=g))
        print("VAE ...");  record("vae", train_vae(members, args).sample(len(members)))
        for eps in [float(e) for e in args.epsilons.split(",")]:
            print(f"VAE + DP (eps={eps:g}) ...")
            record("vae_dp", train_vae(members, args, eps).sample(len(members)), eps)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "synth_per_seed.csv", index=False, float_format="%.4f")
    summ = (res.fillna({"epsilon": -1}).groupby(["setting", "epsilon"], sort=False)
               [["attack_auc", "tstr_mae", "fidelity", "trtr_mae", "naive_mae"]]
               .agg(["mean", "min", "max"]).reset_index())
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ["epsilon"] = summ["epsilon"].replace(-1, np.nan)
    summ.to_csv(out / "synth.csv", index=False, float_format="%.3f")
    (out / "synth_config.json").write_text(json.dumps(vars(args) | {"source": source}, indent=2))

    print("\n=== mean over seeds ===")
    print(f"  {'setting':<14} | {'attack AUC':>10} | {'TSTR MAE':>8} | {'fidelity W1':>11}")
    for r in summ.itertuples():
        name = r.setting + ("" if pd.isna(r.epsilon) else f" eps={r.epsilon:g}")
        print(f"  {name:<14} | {r.attack_auc_mean:10.3f} | {r.tstr_mae_mean:8.3f} | {r.fidelity_mean:11.3f}")
    r0 = summ.iloc[0]
    print(f"  reference: trained on real members MAE={r0.trtr_mae_mean:.3f}, naive (next = previous) MAE={r0.naive_mae_mean:.3f}")
    plot(summ, out / "synth.png", source)
    print(f"\nSaved {out}/synth.csv and {out}/synth.png")


def plot(summ, path, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def label(r):
        if r.setting == "real_noise":
            return "Real data + noise"
        if r.setting == "vae":
            return "VAE (no DP)"
        return f"VAE + DP ε={r.epsilon:g}"

    colors = {"real_noise": "#e07a5f", "vae": "#2a6f97", "vae_dp": "#81b29a"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axvline(0.5, color="#444", lw=0.8, ls="--")
    ax.text(0.502, 0.02, "chance (no leakage)", transform=ax.get_xaxis_transform(), fontsize=7.5, color="#444")
    ref, naive = summ.trtr_mae_mean.iloc[0], summ.naive_mae_mean.iloc[0]
    ax.axhline(ref, color="#9c89b8", lw=0.8, ls=":")
    ax.text(0.01, ref, "trained on real data", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=7.5, color="#9c89b8")
    ax.axhline(naive, color="#999", lw=0.8, ls=":")
    ax.text(0.01, naive, "naive (next day = previous day)", transform=ax.get_yaxis_transform(), ha="left",
            va="bottom", fontsize=7.5, color="#777")
    for r in summ.itertuples():
        xe = [[r.attack_auc_mean - r.attack_auc_min], [r.attack_auc_max - r.attack_auc_mean]]
        ye = [[r.tstr_mae_mean - r.tstr_mae_min], [r.tstr_mae_max - r.tstr_mae_mean]]
        ax.errorbar(r.attack_auc_mean, r.tstr_mae_mean, xerr=xe, yerr=ye, fmt="o", ms=7,
                    color=colors[r.setting], ecolor="#888", lw=0.8, capsize=2)
        ax.annotate(label(r), (r.attack_auc_mean, r.tstr_mae_mean), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("DCR membership inference AUC on the released data (lower = more private)", fontsize=9)
    ax.set_ylabel("Train-synthetic-test-real MAE (normalised, lower = more useful)", fontsize=9)
    ax.set_title(f"Releasing synthetic load data: privacy vs utility ({source})", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
