"""
Empirical privacy audit: membership inference against the federated forecasters.

For every client, the training-period days are split at random into members (used
for training) and held-out non-members from the same period. Each setting is then
trained on the members only, and a loss-threshold membership inference attack tries
to tell members from non-members using the model the attacker can see:

  local          - the client's own model (worst case: the model leaks)
  local_dp       - the client's own model trained with DP-SGD, one run per epsilon
  central        - the model trained on pooled data
  fedavg         - the shared global model (what the server and all clients see)
  fedavg_dp      - the shared global model trained with DP-SGD, one run per epsilon

Attack score: a day is predicted "member" when the model's pinball loss on it is low.
Attack success is reported as AUC (0.5 = no better than chance, 1.0 = perfect), per
client and then averaged, separately for new and full-history clients. Utility is the
pinball loss on the test period.

Usage (same data and options as fl_energy.py):
  python mia.py                       # UCI data
  python mia.py --synthetic
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fl_energy as F


def split_members(clients, seed):
    """Split each client's training days at random into members and non-members."""
    rng = np.random.default_rng(seed + 123)
    out = []
    for c in clients:
        n = len(c["Xtr"])
        idx = torch.tensor(rng.permutation(n))
        m, h = idx[: n // 2], idx[n // 2 :]
        d = dict(c)
        d.update(Xtr=c["Xtr"][m], ytr=c["ytr"][m], Xnon=c["Xtr"][h], ynon=c["ytr"][h])
        out.append(d)
    return out


def per_day_loss(model, X, y):
    model.eval()
    with torch.no_grad():
        q = model(X).view(-1, 24, F.NQ)
    diff = y.unsqueeze(-1) - q
    return torch.maximum(F.QUANTILES * diff, (F.QUANTILES - 1) * diff).mean(dim=(1, 2)).numpy()


def auc(pos, neg):
    """AUC of scores where higher means 'member' (Mann-Whitney U)."""
    s = np.concatenate([pos, neg])
    ranks = pd.Series(s).rank().values
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def attack(models, clients):
    """Per-client loss-threshold attack AUC (lower loss -> member)."""
    return [auc(-per_day_loss(m, c["Xtr"], c["ytr"]), -per_day_loss(m, c["Xnon"], c["ynon"]))
            for m, c in zip(models, clients)]


def fedavg_with_updates(clients, args):
    """FedAvg (as in fl_energy.run_fedavg) that also returns each client's last local
    model, i.e. the update an honest-but-curious server receives in the final round."""
    import copy
    torch.manual_seed(args.seed)
    gm = F.make_model()
    loaders = [F.loader(c, args.batch) for c in clients]
    weights = [len(c["Xtr"]) for c in clients]
    for _ in range(args.rounds):
        locals_ = []
        for dl in loaders:
            local = copy.deepcopy(gm)
            F.train_epochs(local, dl, args.local_epochs, args.fl_lr, sgd=True, clip=args.clip)
            locals_.append(local)
        gm.load_state_dict(F.fedavg([m.state_dict() for m in locals_], weights))
    return gm, locals_


def fedavg_dp_with_updates(clients, args, epsilon):
    """FedAvg + DP-SGD (as in fl_energy.run_fedavg_dp), also returning the last local models."""
    import copy
    from opacus import PrivacyEngine
    torch.manual_seed(args.seed)
    gm = F.make_model()
    privates = []
    for c in clients:
        model = F.make_model()
        opt = torch.optim.SGD(model.parameters(), lr=args.dp_lr, momentum=0.9)
        pe = PrivacyEngine(accountant="rdp")
        model, opt, dl = pe.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=F.loader(c, args.dp_batch),
            target_epsilon=epsilon, target_delta=1.0 / len(c["Xtr"]),
            epochs=args.rounds * args.local_epochs, max_grad_norm=args.clip)
        privates.append(dict(model=model, opt=opt, dl=dl))
    weights = [len(c["Xtr"]) for c in clients]
    for _ in range(args.rounds):
        locals_ = []
        for p in privates:
            p["model"]._module.load_state_dict(gm.state_dict())
            p["model"].train()
            for _ in range(args.local_epochs):
                for xb, yb in p["dl"]:
                    if len(xb) == 0:
                        continue
                    p["opt"].zero_grad()
                    F.pinball_loss(p["model"](xb), yb).backward()
                    p["opt"].step()
            m = F.make_model()
            m.load_state_dict(copy.deepcopy(p["model"]._module.state_dict()))
            locals_.append(m)
        gm.load_state_dict(F.fedavg([m.state_dict() for m in locals_], weights))
    return gm, locals_


def local_dp(clients, args, epsilon):
    """Each client trains alone with DP-SGD for the same number of epochs as local training."""
    from opacus import PrivacyEngine
    models = []
    for c in clients:
        torch.manual_seed(args.seed)
        model = F.make_model()
        opt = torch.optim.SGD(model.parameters(), lr=args.local_dp_lr, momentum=0.9)
        pe = PrivacyEngine(accountant="rdp")
        epochs = args.rounds * args.local_epochs
        model, opt, dl = pe.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=F.loader(c, args.dp_batch),
            target_epsilon=epsilon, target_delta=1.0 / len(c["Xtr"]),
            epochs=epochs, max_grad_norm=args.clip)
        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                if len(xb) == 0:
                    continue
                opt.zero_grad()
                F.pinball_loss(model(xb), yb).backward()
                opt.step()
        m = F.make_model()
        m.load_state_dict(model._module.state_dict())
        models.append(m)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="LD2011_2014.txt")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dp-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fl-lr", type=float, default=0.05)
    ap.add_argument("--dp-lr", type=float, default=0.1)
    ap.add_argument("--local-dp-lr", type=float, default=0.003,
                    help="SGD lr for DP-SGD on a single client (tiny datasets need a smaller step)")
    ap.add_argument("--pers-epochs", type=int, default=5)
    ap.add_argument("--pers-lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epsilons", default="1,4,8")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--scarce-frac", type=float, default=0.5)
    ap.add_argument("--scarce-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0, help="seed for choosing clients")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot-only", action="store_true", help="redraw mia.png from <out>/mia.csv")
    args = ap.parse_args()
    if args.plot_only:
        return replot(args.out)

    if args.synthetic:
        df, source = F.make_synthetic(args.clients, args.seed), "synthetic"
    else:
        df, source = F.load_uci(args.data, args.clients, args.seed), "UCI ElectricityLoadDiagrams20112014"
    n_scarce = int(round(args.scarce_frac * len(df.columns)))
    base = [F.client_arrays(df[col], args.test_frac, args.scarce_days if i < n_scarce else None)
            for i, col in enumerate(df.columns)]
    scarce = np.array([c["scarce"] for c in base])

    def grp(v):
        v = np.array(v)
        return float(v[~scarce].mean()), float(v[scarce].mean())

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        args.seed = seed
        clients = split_members(base, seed)
        print(f"\n=== seed {seed} | {source} | members per full client: {len(clients[-1]['Xtr'])}, "
              f"per new client: {len(clients[0]['Xtr'])} ===")

        def record(name, models, eps=None, updates=None):
            a_full, a_new = grp(attack(models, clients))
            u_a_full, u_a_new = grp(attack(updates, clients)) if updates is not None else (np.nan, np.nan)
            ev = F.evaluate(models, clients, 0.0, seed)
            u_full, u_new = grp([e["pinball"] for e in ev])
            rows.append(dict(seed=seed, setting=name, epsilon=eps, auc_full=a_full, auc_new=a_new,
                             upd_auc_full=u_a_full, upd_auc_new=u_a_new,
                             pinball_full=u_full, pinball_new=u_new))
            upd = (f" | on client updates: full={u_a_full:.3f} new={u_a_new:.3f}"
                   if updates is not None else "")
            print(f"  {name:<10} eps={str(eps):<4} attack AUC full={a_full:.3f} new={a_new:.3f}{upd} | "
                  f"test pinball full={u_full:.2f}% new={u_new:.2f}%")

        print("Local only ...");   record("local", F.run_local(clients, args))
        for eps in [float(e) for e in args.epsilons.split(",")]:
            print(f"Local + DP (eps={eps:g}) ...")
            record("local_dp", local_dp(clients, args, eps), eps)
        print("Centralized ..."); record("central", F.run_central(clients, args))
        print("FedAvg ...")
        gm, ups = fedavg_with_updates(clients, args)
        record("fedavg", [gm] * len(clients), updates=ups)
        for eps in [float(e) for e in args.epsilons.split(",")]:
            print(f"FedAvg + DP (eps={eps:g}) ...")
            gm, ups = fedavg_dp_with_updates(clients, args, eps)
            record("fedavg_dp", [gm] * len(clients), eps, updates=ups)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "mia_per_seed.csv", index=False, float_format="%.4f")
    summ = (res.fillna({"epsilon": -1}).groupby(["setting", "epsilon"], sort=False)
               [["auc_full", "auc_new", "upd_auc_full", "upd_auc_new", "pinball_full", "pinball_new"]]
               .agg(["mean", "min", "max"]).reset_index())
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ["epsilon"] = summ["epsilon"].replace(-1, np.nan)
    summ.to_csv(out / "mia.csv", index=False, float_format="%.3f")
    (out / "mia_config.json").write_text(json.dumps(vars(args) | {"source": source}, indent=2))

    print("\n=== mean over seeds ===")
    print(f"  {'setting':<16} | {'shared/own model AUC':>21} | {'client-update AUC':>18} | {'test pinball':>14}")
    print(f"  {'':<16} | {'full':>10} {'new':>10} | {'full':>8} {'new':>9} | {'full':>6} {'new':>7}")
    for r in summ.itertuples():
        name = r.setting + ("" if pd.isna(r.epsilon) else f" eps={r.epsilon:g}")
        print(f"  {name:<16} | {r.auc_full_mean:10.3f} {r.auc_new_mean:10.3f} | "
              f"{r.upd_auc_full_mean:8.3f} {r.upd_auc_new_mean:9.3f} | "
              f"{r.pinball_full_mean:5.2f}% {r.pinball_new_mean:6.2f}%")
    plot(summ, out / "mia.png", source, len(args.seeds.split(",")))
    print(f"\nSaved {out}/mia.csv and {out}/mia.png")


def plot(summ, path, source, n_seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def label(r):
        names = {"local": "Local only", "central": "Centralized", "fedavg": "FedAvg"}
        if r.setting == "local_dp":
            return f"Local+DP ε={r.epsilon:g}"
        return names.get(r.setting, f"FedAvg+DP ε={r.epsilon:g}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"local": "#e07a5f", "local_dp": "#f2b8a2", "central": "#9c89b8", "fedavg": "#2a6f97", "fedavg_dp": "#81b29a"}
    for ax, g, title in [(axes[0], "new", "New clients (30 days of history)"),
                         (axes[1], "full", "Clients with full history")]:
        ys = summ[f"pinball_{g}_mean"].values
        ok = ys[ys < 50]
        cap = max(ok.max() * 1.35, 1.0)
        ax.set_ylim(0, cap * 1.08)
        ax.axvline(0.5, color="#444", lw=0.8, ls="--")
        ax.text(0.502, 0.02, "chance (no leakage)", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color="#444", va="bottom")
        pts = []
        for r in summ.itertuples():
            x, y = getattr(r, f"auc_{g}_mean"), getattr(r, f"pinball_{g}_mean")
            xe = [[x - getattr(r, f"auc_{g}_min")], [getattr(r, f"auc_{g}_max") - x]]
            off = y > cap  # model unusable: draw at the top edge
            ax.errorbar(x, cap if off else y, xerr=xe, fmt="^" if off else "o", ms=8 if off else 7,
                        color=colors[r.setting], ecolor="#888", lw=0.8, capsize=2)
            pts.append([x, cap if off else y, label(r) + (f" (off scale: {y:,.0f}%)" if off else "")])
        # one label for points that coincide (e.g. Local+DP at every ε on full clients)
        merged = []
        for x, y, lab in pts:
            for m in merged:
                if abs(m[0] - x) < 0.002 and abs(m[1] - y) < 0.02 * cap:
                    m[2].append(lab)
                    break
            else:
                merged.append([x, y, [lab]])
        # stack labels of nearby points so they do not overlap
        fig.canvas.draw()
        to_px = ax.transData.transform
        placed = []
        for x, y, labs in sorted(merged, key=lambda m: -m[1]):
            if len(labs) > 1 and all(l.split(" ε=")[0] == labs[0].split(" ε=")[0] for l in labs):
                text = labs[0].split(" ε=")[0] + " ε=" + ", ".join(l.split(" ε=")[1] for l in labs)
            else:
                text = " / ".join(labs)
            px, py = to_px((x, y))
            dy = 4
            while any(abs(px - qx) < 90 and abs((py + dy * 1.6) - qy) < 11 for qx, qy in placed):
                dy -= 7
            placed.append((px, py + dy * 1.6))
            ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, dy), fontsize=7.5,
                        arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.5) if dy != 4 else None)
        ax.set_xlabel("Membership inference attack AUC (lower = more private)", fontsize=9)
        ax.set_ylabel("Test pinball loss, % of mean load (lower = more useful)", fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Privacy-utility audit of day-ahead load forecasters ({source}); "
                 f"mean and min–max over {n_seeds} seed{'s' if n_seeds != 1 else ''}", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def replot(out):
    """Redraw mia.png from a saved mia.csv, without retraining."""
    out = Path(out)
    summ = pd.read_csv(out / "mia.csv")
    cfg = json.loads((out / "mia_config.json").read_text())
    plot(summ, out / "mia.png", cfg["source"], len(str(cfg["seeds"]).split(",")))
    print(f"Redrew {out}/mia.png")


if __name__ == "__main__":
    main()
