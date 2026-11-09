"""
Secure aggregation for the federated forecasters (simplified Bonawitz et al., CCS 2017).

Question: the privacy audit (mia.py) showed that the update each client sends to the
server can reveal which days it was trained on, and that DP-SGD updates from data-poor
clients leak more than non-private ones. Can we stop the server from seeing individual
updates at all, without changing the model?

Protocol (honest-but-curious server, clients do not collude with it):
  1. Setup: every client creates an X25519 key pair and publishes its public key.
     Each pair of clients (i, j) derives a shared secret by Diffie-Hellman.
  2. Each round, client i encodes its weighted update  n_i * theta_i  in fixed point
     (integers mod 2^64) and appends its weight n_i. It then adds a pseudorandom mask
     PRG(s_ij, round) for every j > i and subtracts it for every j < i.
  3. The server adds the masked vectors. Every mask appears once with + and once
     with -, so they cancel: the server learns sum_i n_i * theta_i and sum_i n_i,
     hence the FedAvg average, but no individual update and no individual weight.
  4. Dropout: if some clients do not send, the survivors reveal the pairwise seeds
     they share with the dropped clients, and the server removes those masks.

Simplifications versus the full protocol: no double masking or Shamir sharing of keys
(so a late-arriving update from a "dropped" client could be unmasked), no malicious
server, and O(n^2) key agreement. They are listed in the README.

Experiments (same data, clients and options as mia.py):
  A. Correctness: the SecAgg global model equals the plain FedAvg model (up to
     fixed-point rounding) and forecasts equally well.
  B. Privacy: the loss-threshold membership attack of mia.py, run on the updates the
     server sees, with and without SecAgg, for FedAvg and FedAvg + DP-SGD.
  C. Dropout: 2 of 20 clients drop in the last round; the recovered average equals
     the plain average of the survivors.
  D. Cost: payload per client and time per round.

Usage:
  python secagg.py                 # UCI data (LD2011_2014.txt in this folder)
  python secagg.py --synthetic
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import fl_energy as F
import mia as M

SCALE = 2 ** 24  # fixed-point scale: resolution about 6e-8


def load_uci_hourly(path, n_clients, seed):
    """Same UCI data as hourly mirror (electricity.txt from the LSTNet benchmark: 321 clients,
    hourly totals from 2012-01-01 01:00, no header). Used when the original archive is not
    available; client identities are column indices."""
    raw = pd.read_csv(path, header=None)
    raw.index = pd.date_range("2012-01-01 01:00", periods=len(raw), freq="1h")
    raw.columns = [f"C{i:03d}" for i in raw.columns]
    hourly = raw.loc["2013-01-01":"2014-12-31"]
    active = hourly.columns[(hourly == 0).mean() < 0.01]
    rng = np.random.default_rng(seed)
    cols = rng.choice(active, size=min(n_clients, len(active)), replace=False)
    return hourly[sorted(cols)]



# ----------------------------------------------------------------------------
# Encoding: float vectors <-> integers mod 2^64 (numpy uint64 wraps around)
# ----------------------------------------------------------------------------
def encode(x):
    return np.round(np.asarray(x, dtype=np.float64) * SCALE).astype(np.int64).view(np.uint64)


def decode(u):
    return u.view(np.int64).astype(np.float64) / SCALE


def flatten(state):
    return np.concatenate([v.detach().cpu().numpy().ravel() for v in state.values()])


def unflatten(vec, like):
    out, i = {}, 0
    for k, v in like.items():
        n = v.numel()
        out[k] = torch.tensor(vec[i:i + n], dtype=v.dtype).view_as(v)
        i += n
    return out


# ----------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------
class Client:
    def __init__(self, cid):
        self.cid = cid
        self._sk = X25519PrivateKey.generate()
        self.public = self._sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)  # 32 bytes
        self.secrets = {}

    def agree(self, publics):
        """Derive a shared secret with every other client from the published keys."""
        for j, pk in publics.items():
            if j != self.cid:
                self.secrets[j] = self._sk.exchange(X25519PublicKey.from_public_bytes(pk))

    def mask_seed(self, j, rnd):
        lo, hi = sorted((self.cid, j))
        info = f"secagg|round={rnd}|pair={lo},{hi}".encode()
        key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(self.secrets[j])
        return int.from_bytes(key, "big")

    def masked_input(self, vec, rnd):
        y = encode(vec)
        for j in self.secrets:
            m = prg(self.mask_seed(j, rnd), len(y))
            y = y + m if self.cid < j else y - m
        return y

    def reveal_seeds(self, dropped, rnd):
        """Dropout recovery: seeds shared with clients that did not send this round."""
        return {j: self.mask_seed(j, rnd) for j in dropped}


def prg(seed, n):
    return np.random.Generator(np.random.PCG64(seed)).integers(0, 2 ** 64, size=n, dtype=np.uint64,
                                                              endpoint=False)


def server_sum(masked, survivors, dropped, clients, rnd):
    """Sum the masked vectors of the survivors and remove masks shared with dropped clients."""
    total = np.zeros_like(masked[survivors[0]])
    for i in survivors:
        total = total + masked[i]
    for i in survivors:
        for j, seed in clients[i].reveal_seeds(dropped, rnd).items():
            m = prg(seed, len(total))
            total = total - m if i < j else total + m  # undo what client i added for pair (i, j)
    return decode(total)


class SecureFedAvg:
    """Drop-in replacement for fl_energy.fedavg(states, weights) that aggregates through
    the masking protocol. Keeps what the server saw in the last round."""

    def __init__(self, n_clients):
        self.clients = [Client(i) for i in range(n_clients)]
        publics = {c.cid: c.public for c in self.clients}
        for c in self.clients:
            c.agree(publics)
        self.round = 0
        self.last_masked = None
        self.times = []

    def __call__(self, states, weights, dropped=()):
        t0 = time.perf_counter()
        like = states[0]
        masked = {}
        for c, s, w in zip(self.clients, states, weights):
            if c.cid in dropped:
                continue
            vec = np.append(flatten(s).astype(np.float64) * w, w)  # weighted update + weight
            masked[c.cid] = c.masked_input(vec, self.round)
        survivors = sorted(masked)
        agg = server_sum(masked, survivors, list(dropped), self.clients, self.round)
        avg = agg[:-1] / agg[-1]
        self.last_masked = masked
        self.round += 1
        self.times.append(time.perf_counter() - t0)
        return unflatten(avg.astype(np.float32), like)


def server_view_models(masked, weights_like_state):
    """Turn each masked vector into a model, as a curious server would try to."""
    models = []
    for i in sorted(masked):
        v = decode(masked[i])[:-1]
        v = np.clip(v, -1e6, 1e6).astype(np.float32)  # values are noise of size ~2^39
        m = F.make_model()
        m.load_state_dict(unflatten(v, weights_like_state))
        models.append(m)
    return models


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def with_aggregator(agg, fn, *a):
    """Run an existing FedAvg routine from mia.py with fl_energy.fedavg replaced by agg."""
    orig = F.fedavg
    F.fedavg = agg
    try:
        return fn(*a)
    finally:
        F.fedavg = orig


def max_param_diff(m1, m2):
    return max(float((a - b).abs().max()) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="LD2011_2014.txt")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--hourly", default=None, help="hourly mirror electricity.txt instead of the original archive")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dp-batch", type=int, default=256)
    ap.add_argument("--fl-lr", type=float, default=0.05)
    ap.add_argument("--dp-lr", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=8.0, help="DP-SGD budget for the FedAvg+DP runs")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--scarce-frac", type=float, default=0.5)
    ap.add_argument("--scarce-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0, help="seed for choosing clients")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.synthetic:
        df, source = F.make_synthetic(args.clients, args.seed), "synthetic"
    elif args.hourly:
        df, source = load_uci_hourly(args.hourly, args.clients, args.seed), "UCI electricity (hourly mirror)"
    else:
        df, source = F.load_uci(args.data, args.clients, args.seed), "UCI ElectricityLoadDiagrams20112014"
    n_scarce = int(round(args.scarce_frac * len(df.columns)))
    base = [F.client_arrays(df[col], args.test_frac, args.scarce_days if i < n_scarce else None)
            for i, col in enumerate(df.columns)]
    scarce = np.array([c["scarce"] for c in base])

    def grp(v):
        v = np.array(v)
        return float(v[~scarce].mean()), float(v[scarce].mean())

    rows, checks = [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        args.seed = seed
        clients = M.split_members(base, seed)
        n = len(clients)
        print(f"\n=== seed {seed} | {source} | {n} clients ===")

        def record(name, gm, updates, server_view=None):
            ev = F.evaluate([gm] * n, clients, 0.0, seed)
            p_full, p_new = grp([e["pinball"] for e in ev])
            u_full, u_new = grp(M.attack(updates, clients))
            row = dict(seed=seed, setting=name, pinball_full=p_full, pinball_new=p_new,
                       upd_auc_full=u_full, upd_auc_new=u_new)
            if server_view is not None:
                s_full, s_new = grp(M.attack(server_view, clients))
                row.update(server_auc_full=s_full, server_auc_new=s_new)
            rows.append(row)
            sv = (f" | attack on what the server sees: full={row['server_auc_full']:.3f} "
                  f"new={row['server_auc_new']:.3f}" if server_view is not None else
                  f" | attack on what the server sees: same as left")
            print(f"  {name:<18} test pinball full={p_full:.2f}% new={p_new:.2f}% | "
                  f"attack on true updates: full={u_full:.3f} new={u_new:.3f}{sv}")

        # A + B: FedAvg, plain vs secure aggregation
        gm_p, ups_p = M.fedavg_with_updates(clients, args)
        record("FedAvg", gm_p, ups_p)
        agg = SecureFedAvg(n)
        gm_s, ups_s = with_aggregator(agg, M.fedavg_with_updates, clients, args)
        view = server_view_models(agg.last_masked, gm_s.state_dict())
        record("FedAvg + SecAgg", gm_s, ups_s, view)
        true_last = np.append(flatten(ups_s[0].state_dict()) * len(clients[0]["Xtr"]), len(clients[0]["Xtr"]))
        c_corr = corr(decode(agg.last_masked[0]), true_last)
        diff = max_param_diff(gm_p, gm_s)
        sec_time = float(np.mean(agg.times))

        # FedAvg + DP-SGD, plain vs secure aggregation
        gm_dp, ups_dp = M.fedavg_dp_with_updates(clients, args, args.epsilon)
        record(f"FedAvg + DP e={args.epsilon:g}", gm_dp, ups_dp)
        agg_dp = SecureFedAvg(n)
        gm_dps, ups_dps = with_aggregator(agg_dp, M.fedavg_dp_with_updates, clients, args, args.epsilon)
        view_dp = server_view_models(agg_dp.last_masked, gm_dps.state_dict())
        record(f"FedAvg + DP e={args.epsilon:g} + SecAgg", gm_dps, ups_dps, view_dp)

        # C: dropout of 2 clients, checked on the final local models of the plain run
        states = [m.state_dict() for m in ups_p]
        weights = [len(c["Xtr"]) for c in clients]
        dropped = (3, 11)
        surv = [i for i in range(n) if i not in dropped]
        agg_d = SecureFedAvg(n)
        rec = agg_d(states, weights, dropped=dropped)
        ref = F.fedavg([states[i] for i in surv], [weights[i] for i in surv])
        drop_diff = max(float((rec[k] - ref[k]).abs().max()) for k in ref)

        d = len(flatten(states[0])) + 1
        checks.append(dict(seed=seed, max_param_diff_vs_fedavg=diff, dropout_max_diff=drop_diff,
                           corr_masked_vs_true_update=c_corr, params=d,
                           payload_plain_kB=4 * (d - 1) / 1e3, payload_secagg_kB=8 * d / 1e3,
                           public_key_bytes=32, secagg_seconds_per_round=sec_time))
        print(f"  check: max |SecAgg - FedAvg| parameter difference = {diff:.2e}")
        print(f"  check: dropout of clients {dropped}: max difference from survivors' FedAvg = {drop_diff:.2e}")
        print(f"  check: correlation between a masked vector and the true update = {c_corr:+.3f}")
        print(f"  cost : {d} values, payload {4*(d-1)/1e3:.0f} kB (float32) -> {8*d/1e3:.0f} kB (uint64), "
              f"+32-byte public key; SecAgg {sec_time*1e3:.0f} ms per round for {n} clients")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "secagg_per_seed.csv", index=False, float_format="%.4f")
    summ = res.drop(columns="seed").groupby("setting", sort=False).agg(["mean", "min", "max"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "secagg.csv", index=False, float_format="%.4f")
    pd.DataFrame(checks).to_csv(out / "secagg_checks.csv", index=False, float_format="%.6g")
    (out / "secagg_config.json").write_text(json.dumps(vars(args) | {"source": source, "scale": SCALE}, indent=2))

    print("\n=== mean over seeds ===")
    print(f"  {'setting':<26} | {'test pinball':>14} | {'attack on true updates':>22} | {'attack on server view':>22}")
    print(f"  {'':<26} | {'full':>6} {'new':>7} | {'full':>10} {'new':>11} | {'full':>10} {'new':>11}")
    for r in summ.itertuples():
        sf = getattr(r, "server_auc_full_mean", np.nan)
        sn = getattr(r, "server_auc_new_mean", np.nan)
        if pd.isna(sf):
            sf, sn = r.upd_auc_full_mean, r.upd_auc_new_mean
        print(f"  {r.setting:<26} | {r.pinball_full_mean:5.2f}% {r.pinball_new_mean:6.2f}% | "
              f"{r.upd_auc_full_mean:10.3f} {r.upd_auc_new_mean:11.3f} | {sf:10.3f} {sn:11.3f}")
    plot(summ, out / "secagg.png", source, len(args.seeds.split(",")))
    print(f"\nSaved {out}/secagg.csv, {out}/secagg_checks.csv and {out}/secagg.png")


def plot(summ, path, source, n_seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    labels, x = [], np.arange(len(summ))
    for i, r in enumerate(summ.itertuples()):
        sec = "SecAgg" in r.setting
        v = r.server_auc_new_mean if sec else r.upd_auc_new_mean
        lo = (r.server_auc_new_min if sec else r.upd_auc_new_min)
        hi = (r.server_auc_new_max if sec else r.upd_auc_new_max)
        ax.bar(i, v - 0.5, bottom=0.5, color="#2a6f97" if sec else "#e07a5f", width=0.6)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="#444", capsize=3, lw=0.8)
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        labels.append(r.setting.replace(" + SecAgg", "\n+ SecAgg"))
    ax.axhline(0.5, color="#444", lw=0.8, ls="--")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("Membership attack AUC on what\nthe server sees (new clients)", fontsize=9)
    ax.set_ylim(0.45, max(0.62, float(summ["upd_auc_new_max"].max()) + 0.02))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Server-side membership inference, with and without secure aggregation\n"
                 f"({source}; mean of {n_seeds} seeds; 0.5 = chance)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
