"""
Privacy-preserving federated day-ahead load forecasting, with a battery-scheduling
decision layer.

Each electricity consumer is one federated client. Raw consumption data never leaves
the client: clients train locally and share only model weights (FedAvg). Optionally,
each client trains with DP-SGD (Opacus), giving record-level (epsilon, delta)-DP for
its own data.

Forecast: at midnight, predict the 24 hourly loads of the next day, as three quantiles
(10%, 50%, 90%), from the past 7 days of hourly readings and calendar features.

Decision: each consumer has a battery and a time-of-use tariff. The median forecast is
used to plan the next day's charge/discharge schedule (a linear programme). The plan is
then applied to the actual load, and we measure how much of the savings that a perfect
forecast would achieve is actually captured.

Compared settings:
  naive_day / naive_week - yesterday's / last week's profile (standard references)
  local                  - each client trains alone
  central                - all data pooled (upper bound, not privacy-preserving)
  fedavg                 - federated averaging, no DP
  fedavg_dp              - federated averaging with DP-SGD on every client
  *_pers                 - each client fine-tunes the global model on its own data
                           (the personalised model never leaves the client)

Robustness: forecasts are also evaluated with a fraction of input readings missing.

Data: UCI "ElectricityLoadDiagrams20112014" (LD2011_2014.txt)
https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
Use --synthetic to run without the dataset.
"""
import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

HIST_DAYS = 7                  # input: the past 7 days of hourly readings
N_IN = 24 * HIST_DAYS          # input columns that are meter readings (can go missing)
N_CAL = 4                      # calendar features of the target day
QUANTILES = torch.tensor([0.1, 0.5, 0.9])
NQ = len(QUANTILES)

# Time-of-use tariff (EUR/kWh) by hour of day: off-peak, mid, peak
PRICE = np.array([0.10] * 7 + [0.20] * 11 + [0.32] * 4 + [0.20] * 2)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_uci(path, n_clients, seed):
    df = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.loc["2013-01-01":"2014-12-31"]
    hourly = df.resample("1h").sum() / 4.0  # 15-min kW readings -> kWh per hour
    active = hourly.columns[(hourly == 0).mean() < 0.01]  # active over the whole period
    rng = np.random.default_rng(seed)
    cols = rng.choice(active, size=min(n_clients, len(active)), replace=False)
    return hourly[sorted(cols)]


def make_synthetic(n_clients, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2013-01-01", "2014-12-31 23:00", freq="1h")
    h, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values
    data = {}
    for c in range(n_clients):
        scale = rng.uniform(5, 200)
        kind = rng.integers(0, 3)  # residential / office / industrial-like profiles
        if kind == 0:
            daily = 0.6 + 0.4 * np.exp(-((h - 19) ** 2) / 8) + 0.2 * np.exp(-((h - 8) ** 2) / 4)
            weekly = np.ones_like(h, dtype=float)
        elif kind == 1:
            daily = 0.3 + 0.7 * ((h >= 8) & (h <= 18))
            weekly = np.where(dow >= 5, 0.4, 1.0)
        else:
            daily = 0.8 + 0.2 * np.sin(2 * np.pi * h / 24)
            weekly = np.where(dow == 6, 0.7, 1.0)
        seasonal = 1 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365)
        day_level = np.repeat(rng.normal(1, 0.06, len(idx) // 24), 24)  # day-to-day variation
        noise = rng.normal(1, 0.08, len(idx))
        data[f"MT_{c:03d}"] = scale * daily * weekly * seasonal * day_level * noise
    return pd.DataFrame(data, index=idx)


def client_arrays(series, test_frac, scarce_days=None):
    """One example per day: inputs = past 7 days (168 h) + calendar of the target day,
    target = the 24 hourly loads of the target day.
    Each client normalises with statistics from its own training history only.
    New clients (scarce_days) see only their last `scarce_days` days of history: every
    input and target of their training examples falls inside that window."""
    y = series.values.astype(np.float32)
    n_days = len(y) // 24
    y = y[: n_days * 24]
    split_day = int(n_days * (1 - test_frac))
    lo_day = 0 if scarce_days is None else max(0, split_day - scarce_days)
    hist = y[lo_day * 24 : split_day * 24]
    mu, sd = hist.mean(), hist.std() + 1e-6
    z = (y - mu) / sd
    days = series.index[: n_days * 24 : 24]
    X, T = [], []
    for k in range(HIST_DAYS, n_days):
        d = days[k]
        cal = [np.sin(2 * np.pi * d.dayofweek / 7), np.cos(2 * np.pi * d.dayofweek / 7),
               np.sin(2 * np.pi * d.dayofyear / 365.25), np.cos(2 * np.pi * d.dayofyear / 365.25)]
        X.append(np.concatenate([z[(k - HIST_DAYS) * 24 : k * 24], cal]))
        T.append(z[k * 24 : (k + 1) * 24])
    X, T = np.array(X, dtype=np.float32), np.array(T, dtype=np.float32)
    split = split_day - HIST_DAYS            # index of the first test day
    start = max(0, lo_day) if scarce_days is not None else 0
    return dict(
        scarce=scarce_days is not None,
        Xtr=torch.tensor(X[start:split]), ytr=torch.tensor(T[start:split]),
        Xte=torch.tensor(X[split:]), yte=torch.tensor(T[split:]),
        mu=float(mu), sd=float(sd),
    )


# ----------------------------------------------------------------------------
# Model and training helpers
# ----------------------------------------------------------------------------
class ResidualOverNaive(nn.Module):
    """Forecast = last week's profile + a learned correction for each hour and quantile.
    The last layer starts at zero, so an untrained model equals the weekly naive forecast;
    learning (and DP noise) only acts on the correction. Small on purpose: under DP-SGD,
    noise is added to every parameter's gradient."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN + N_CAL, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 24 * NQ),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, X):
        anchor = X[:, :24]  # same day last week
        out = self.net(X).view(-1, 24, NQ) + anchor.unsqueeze(-1)
        return out.reshape(len(X), -1)


def make_model():
    return ResidualOverNaive()


def pinball_loss(out, y):
    """Quantile (pinball) loss averaged over hours and quantiles. out: (B, 72), y: (B, 24)."""
    q = out.view(-1, 24, NQ)
    diff = y.unsqueeze(-1) - q
    return torch.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).mean()


def loader(c, batch, shuffle=True):
    return DataLoader(TensorDataset(c["Xtr"], c["ytr"]),
                      batch_size=min(batch, max(4, len(c["Xtr"]) // 2)), shuffle=shuffle)


def train_epochs(model, dl, epochs, lr, sgd=False, clip=None):
    opt = (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) if sgd
           else torch.optim.Adam(model.parameters(), lr=lr))
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            pinball_loss(model(xb), yb).backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()


def fedavg(states, weights):
    total = sum(weights)
    avg = copy.deepcopy(states[0])
    for k in avg:
        avg[k] = sum(s[k] * (w / total) for s, w in zip(states, weights))
    return avg


def personalise(global_model, clients, args):
    """Each client fine-tunes the global model on its own data; the result stays local."""
    models = []
    for c in clients:
        m = copy.deepcopy(global_model)
        train_epochs(m, loader(c, 16), args.pers_epochs, args.pers_lr)
        models.append(m)
    return models


# ----------------------------------------------------------------------------
# Settings: each returns one model per client (the same object for a global model)
# ----------------------------------------------------------------------------
def run_local(clients, args):
    models = []
    for c in clients:
        torch.manual_seed(args.seed)
        m = make_model()
        train_epochs(m, loader(c, args.batch), args.rounds * args.local_epochs, args.lr)
        models.append(m)
    return models


def run_central(clients, args):
    torch.manual_seed(args.seed)
    m = make_model()
    X = torch.cat([c["Xtr"] for c in clients])
    y = torch.cat([c["ytr"] for c in clients])
    dl = DataLoader(TensorDataset(X, y), batch_size=args.batch, shuffle=True)
    train_epochs(m, dl, args.rounds * args.local_epochs, args.fl_lr, sgd=True, clip=args.clip)
    return [m] * len(clients)


def run_fedavg(clients, args):
    torch.manual_seed(args.seed)
    gm = make_model()
    loaders = [loader(c, args.batch) for c in clients]
    weights = [len(c["Xtr"]) for c in clients]
    for _ in range(args.rounds):
        states = []
        for dl in loaders:
            local = copy.deepcopy(gm)
            train_epochs(local, dl, args.local_epochs, args.fl_lr, sgd=True, clip=args.clip)
            states.append(local.state_dict())
        gm.load_state_dict(fedavg(states, weights))
    return gm


def run_fedavg_dp(clients, args, epsilon):
    """FedAvg where every client runs DP-SGD (Opacus). Each client's privacy budget
    covers all its local steps over all rounds: the shared model is (epsilon, delta)-DP
    with respect to any single training day of that client."""
    from opacus import PrivacyEngine

    torch.manual_seed(args.seed)
    gm = make_model()
    privates = []
    for c in clients:
        model = make_model()
        opt = torch.optim.SGD(model.parameters(), lr=args.dp_lr, momentum=0.9)
        pe = PrivacyEngine(accountant="rdp")
        delta = 1.0 / len(c["Xtr"])
        model, opt, dl = pe.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=loader(c, args.dp_batch),
            target_epsilon=epsilon, target_delta=delta,
            epochs=args.rounds * args.local_epochs, max_grad_norm=args.clip,
        )
        privates.append(dict(model=model, opt=opt, dl=dl, pe=pe, delta=delta))

    weights = [len(c["Xtr"]) for c in clients]
    for _ in range(args.rounds):
        states = []
        for p in privates:
            p["model"]._module.load_state_dict(gm.state_dict())
            p["model"].train()
            for _ in range(args.local_epochs):
                for xb, yb in p["dl"]:
                    if len(xb) == 0:  # Poisson sampling can return an empty batch
                        continue
                    p["opt"].zero_grad()
                    pinball_loss(p["model"](xb), yb).backward()
                    p["opt"].step()
            states.append(copy.deepcopy(p["model"]._module.state_dict()))
        gm.load_state_dict(fedavg(states, weights))
    spent = max(p["pe"].get_epsilon(p["delta"]) for p in privates)
    return gm, spent


class Naive(nn.Module):
    """Seasonal naive: repeat yesterday's (lag=1) or last week's (lag=7) profile.
    All three quantiles equal the point forecast (a degenerate distribution)."""
    def __init__(self, lag_days):
        super().__init__()
        self.s = (HIST_DAYS - lag_days) * 24

    def forward(self, X):
        p = X[:, self.s : self.s + 24]
        return p.unsqueeze(-1).repeat(1, 1, NQ).reshape(len(X), -1)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def forecast(model, c, missing_frac, g):
    X = c["Xte"].clone()
    if missing_frac > 0:
        mask = torch.rand(X[:, :N_IN].shape, generator=g) < missing_frac
        X[:, :N_IN][mask] = 0.0  # missing reading -> client mean
    model.eval()
    with torch.no_grad():
        q = model(X).view(-1, 24, NQ)
    q, _ = torch.sort(q, dim=-1)  # no quantile crossing
    q = torch.clamp(q * c["sd"] + c["mu"], min=0.0)  # load cannot be negative
    return q.numpy(), (c["yte"] * c["sd"] + c["mu"]).numpy()


def evaluate(models, clients, missing_frac, seed):
    """Per client: NMAE of the median, pinball loss, and coverage of the 10-90% interval.
    NMAE and pinball are expressed in % of the client's mean test load."""
    g = torch.Generator().manual_seed(seed + 7)
    out = []
    for m, c in zip(models, clients):
        q, y = forecast(m, c, missing_frac, g)
        scale = y.mean()
        diff = y[..., None] - q
        qs = QUANTILES.numpy()
        pin = np.maximum(qs * diff, (qs - 1) * diff).mean()
        out.append(dict(
            nmae=100 * np.abs(q[..., 1] - y).mean() / scale,
            pinball=100 * pin / scale,
            coverage=100 * ((y >= q[..., 0]) & (y <= q[..., 2])).mean(),
            median=q[..., 1], actual=y, q=q,
        ))
    return out


# ----------------------------------------------------------------------------
# Decision layer: battery scheduling with a time-of-use tariff
# ----------------------------------------------------------------------------
def battery_lp(load_forecast, cap, power, eff=0.95):
    """Plan charge c_t and discharge d_t (kWh per hour) to minimise the cost of grid
    imports for the forecast load. No export; state of charge starts and ends at 50%."""
    from scipy.optimize import linprog

    T = 24
    # variables: [c_0..c_23, d_0..d_23, s_1..s_24]
    cost = np.concatenate([PRICE, -PRICE, np.zeros(T)])
    A_eq = np.zeros((T + 1, 3 * T))
    b_eq = np.zeros(T + 1)
    for t in range(T):  # s_t - s_{t-1} - eff*c_t + d_t/eff = 0
        A_eq[t, 2 * T + t] = 1
        if t > 0:
            A_eq[t, 2 * T + t - 1] = -1
        else:
            b_eq[t] = 0.5 * cap
        A_eq[t, t] = -eff
        A_eq[t, T + t] = 1 / eff
    A_eq[T, 3 * T - 1] = 1  # end at 50%
    b_eq[T] = 0.5 * cap
    A_ub = np.zeros((T, 3 * T))  # d_t - c_t <= forecast_t (no export)
    A_ub[np.arange(T), T + np.arange(T)] = 1
    A_ub[np.arange(T), np.arange(T)] = -1
    b_ub = np.maximum(load_forecast, 0)
    bounds = [(0, power)] * (2 * T) + [(0, cap)] * T
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not r.success:
        return np.zeros(T), np.zeros(T)
    return r.x[:T], r.x[T : 2 * T]


def realised_cost(actual, ch, dis):
    # energy discharged beyond the actual load cannot be exported and is lost
    return float((PRICE * np.maximum(actual + ch - dis, 0)).sum())


def battery_reference(clients, actuals):
    """Cost without a battery and with a perfect forecast, per client."""
    ref = []
    for c, ys in zip(clients, actuals):
        cap, power = 8 * c["mu"], 2 * c["mu"]  # battery sized on the client's mean load
        none = float(sum((PRICE * y).sum() for y in ys))
        oracle = 0.0
        for y in ys:
            ch, dis = battery_lp(y, cap, power)
            oracle += realised_cost(y, ch, dis)
        ref.append(dict(none=none, oracle=oracle, cap=cap, power=power))
    return ref


def battery_savings(evals, ref):
    """% of the perfect-forecast savings captured when scheduling on the median forecast."""
    out = []
    for e, r in zip(evals, ref):
        cost = 0.0
        for fc, y in zip(e["median"], e["actual"]):
            ch, dis = battery_lp(fc, r["cap"], r["power"])
            cost += realised_cost(y, ch, dis)
        out.append(100 * (r["none"] - cost) / max(r["none"] - r["oracle"], 1e-9))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="LD2011_2014.txt")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dp-batch", type=int, default=256,
                    help="expected DP-SGD batch (capped at half a client's data); larger batches dilute the noise")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam lr (local, central)")
    ap.add_argument("--fl-lr", type=float, default=0.05, help="SGD lr (FedAvg, with and without DP)")
    ap.add_argument("--dp-lr", type=float, default=0.1,
                    help="SGD lr under DP (per-example clipping shrinks gradients)")
    ap.add_argument("--pers-epochs", type=int, default=5)
    ap.add_argument("--pers-lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epsilons", default="1,4,8")
    ap.add_argument("--missing", default="0,0.2")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--scarce-frac", type=float, default=0.5,
                    help="fraction of clients that are new members with little history")
    ap.add_argument("--scarce-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0, help="seed for choosing clients")
    ap.add_argument("--seeds", default="0,1", help="training seeds (init, shuffling, DP noise, missing mask)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.synthetic:
        df, source = make_synthetic(args.clients, args.seed), "synthetic"
    else:
        df, source = load_uci(args.data, args.clients, args.seed), "UCI ElectricityLoadDiagrams20112014"
    n_scarce = int(round(args.scarce_frac * len(df.columns)))
    clients = [client_arrays(df[col], args.test_frac, args.scarce_days if i < n_scarce else None)
               for i, col in enumerate(df.columns)]
    print(f"Data: {source} | {len(clients)} clients ({n_scarce} new, {len(clients[0]['Xtr'])} training days each) | "
          f"{len(clients[-1]['Xtr'])} training / {len(clients[-1]['Xte'])} test days per full client")

    scarce = np.array([c["scarce"] for c in clients])
    missing_levels = [float(m) for m in args.missing.split(",")]
    actuals = [(c["yte"] * c["sd"] + c["mu"]).numpy() for c in clients]
    print("Battery reference (no battery / perfect forecast) ...")
    ref = battery_reference(clients, actuals)
    rows, examples = [], {}

    def grp(vals):
        v = np.array(vals)
        return (float(v[~scarce].mean()) if (~scarce).any() else np.nan,
                float(v[scarce].mean()) if scarce.any() else np.nan)

    for seed in [int(x) for x in args.seeds.split(",")]:
        args.seed = seed
        print(f"\n=== seed {seed} ===")

        def record(name, models, eps=None):
            for mf in missing_levels:
                ev = evaluate(models, clients, mf, seed)
                row = dict(seed=seed, setting=name, epsilon=eps, missing=mf)
                for k in ["nmae", "pinball", "coverage"]:
                    row[f"{k}_full"], row[f"{k}_new"] = grp([e[k] for e in ev])
                if mf == 0:
                    row["savings_full"], row["savings_new"] = grp(battery_savings(ev, ref))
                    if seed == 0:
                        examples[(name, eps)] = ev
                rows.append(row)
                sav = (f"  battery savings captured: full={row['savings_full']:5.1f}%  new={row['savings_new']:5.1f}%"
                       if mf == 0 else "")
                print(f"  {name:<15} eps={str(eps):<4} missing={mf:.0%}  NMAE full={row['nmae_full']:5.2f}% "
                      f"new={row['nmae_new']:5.2f}%  pinball new={row['pinball_new']:5.2f}%  "
                      f"cov new={row['coverage_new']:5.1f}%{sav}")

        record("naive_day", [Naive(1)] * len(clients))
        record("naive_week", [Naive(7)] * len(clients))
        print("Local only ...");   record("local", run_local(clients, args))
        print("Centralized ..."); record("central", run_central(clients, args))
        print("FedAvg ...")
        gm = run_fedavg(clients, args)
        record("fedavg", [gm] * len(clients))
        record("fedavg_pers", personalise(gm, clients, args))
        for eps in [float(e) for e in args.epsilons.split(",")]:
            print(f"FedAvg + DP (target eps={eps:g}) ...")
            gm, spent = run_fedavg_dp(clients, args, eps)
            record("fedavg_dp", [gm] * len(clients), eps)
            record("fedavg_dp_pers", personalise(gm, clients, args), eps)
            print(f"  (privacy spent: eps={spent:.2f})")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "results_per_seed.csv", index=False, float_format="%.3f")
    metrics = [c for c in res.columns if c.endswith("_full") or c.endswith("_new")]
    summary = (res.fillna({"epsilon": -1})
                  .groupby(["setting", "epsilon", "missing"], sort=False)[metrics]
                  .agg(["mean", "min", "max"]).reset_index())
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary["epsilon"] = summary["epsilon"].replace(-1, np.nan)
    summary.to_csv(out / "results.csv", index=False, float_format="%.2f")

    print("\n=== mean over seeds ===")
    print(f"  {'setting':<22}{'miss':>5} | {'NMAE full':>9} {'NMAE new':>9} | {'pinb new':>8} {'cov new':>8} | "
          f"{'save full':>9} {'save new':>9}")
    for r in summary.itertuples():
        name = r.setting + ("" if pd.isna(r.epsilon) else f" eps={r.epsilon:g}")
        sav = ("" if pd.isna(r.savings_new_mean)
               else f" | {r.savings_full_mean:8.1f}% {r.savings_new_mean:8.1f}%")
        print(f"  {name:<22}{r.missing:>5.0%} | {r.nmae_full_mean:8.2f}% {r.nmae_new_mean:8.2f}% | "
              f"{r.pinball_new_mean:7.2f}% {r.coverage_new_mean:7.1f}%{sav}")

    (out / "config.json").write_text(json.dumps(vars(args) | {"source": source}, indent=2))
    plot_accuracy(summary, out / "results.png", source, len(clients))
    plot_decision(summary, out / "battery.png", source)
    eps_show = min((float(e) for e in args.epsilons.split(",")), key=lambda e: abs(e - 4))
    plot_example(clients, examples, ("fedavg_dp_pers", eps_show), out / "forecast_example.png")
    print(f"\nSaved results to {out}/")


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
LABELS = {"naive_day": "Naive\n(yesterday)", "naive_week": "Naive\n(last week)",
          "local": "Local\nonly", "central": "Centralized\n(no privacy)",
          "fedavg": "FedAvg", "fedavg_pers": "FedAvg\n+pers."}


def label(r):
    if r.setting == "fedavg_dp":
        return f"FedAvg+DP\nε={r.epsilon:g}"
    if r.setting == "fedavg_dp_pers":
        return f"FedAvg+DP\nε={r.epsilon:g}\n+pers."
    return LABELS[r.setting]


def _bars(ax, d, col, x, offset, w, color, lab):
    mean = d[col + "_mean"].values
    err = np.vstack([mean - d[col + "_min"].values, d[col + "_max"].values - mean])
    bars = ax.bar(x + offset, mean, w, yerr=err, capsize=2, error_kw=dict(lw=0.8, ecolor="#444"),
                  label=lab, color=color)
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=3)


def plot_accuracy(summary, path, source, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = sorted(summary.missing.unique())
    base = summary[summary.missing == levels[0]].reset_index(drop=True)
    labels = [label(r) for r in base.itertuples()]
    x, w = np.arange(len(labels)), 0.8 / len(levels)
    colors = ["#2a6f97", "#e9a03b"]
    panels = [("nmae_new", "New clients (30 days of history)"), ("nmae_full", "Clients with full history")]
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        for i, mf in enumerate(levels):
            _bars(ax, summary[summary.missing == mf], col, x, (i - (len(levels) - 1) / 2) * w, w,
                  colors[i % 2], f"{mf:.0%} of inputs missing")
        ax.axvspan(-0.5, 1.5, color="#f2f2f2", zorder=0)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel("Day-ahead NMAE of the median (%)\nlower is better", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(f"Day-ahead load forecasting across {n} clients ({source}); "
                 f"mean of 2 seeds, whiskers show min-max; grey = naive baselines", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_decision(summary, path, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = summary[summary.missing == 0].reset_index(drop=True)
    labels = [label(r) for r in d.itertuples()]
    x, w = np.arange(len(labels)), 0.38
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    _bars(ax, d, "savings_new", x, -w / 2, w, "#2a6f97", "New clients (30 days)")
    _bars(ax, d, "savings_full", x, w / 2, w, "#9ab8cf", "Clients with full history")
    ax.axhline(100, color="#444", lw=0.8, ls="--")
    ax.text(len(labels) - 0.5, 101.5, "perfect forecast", ha="right", va="bottom", fontsize=8, color="#444")
    ax.axvspan(-0.5, 1.5, color="#f2f2f2", zorder=0)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("Battery savings captured (%)\nhigher is better", fontsize=9)
    ax.set_title("Decision value: a battery scheduled on each day-ahead forecast "
                 f"(time-of-use tariff, {source})", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_example(clients, examples, key, path, n_days=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if key not in examples or ("local", None) not in examples:
        return
    i = next((k for k, c in enumerate(clients) if c["scarce"]), 0)
    fed, loc = examples[key][i], examples[("local", None)][i]
    h = np.arange(n_days * 24)
    y = fed["actual"][:n_days].ravel()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(h, fed["q"][:n_days, :, 0].ravel(), fed["q"][:n_days, :, 2].ravel(),
                    color="#2a6f97", alpha=0.2, lw=0, label="FedAvg+DP+pers.: 10-90% interval")
    ax.plot(h, fed["median"][:n_days].ravel(), color="#2a6f97", lw=1.4, label="FedAvg+DP+pers.: median")
    ax.plot(h, loc["median"][:n_days].ravel(), color="#e07a5f", lw=1.0, ls="--", label="Local only: median")
    ax.plot(h, y, color="#222", lw=1.0, label="Actual")
    for d in range(1, n_days):
        ax.axvline(d * 24, color="#ddd", lw=0.6, zorder=0)
    ax.set_xlim(0, n_days * 24)
    ax.set_xticks(np.arange(12, n_days * 24, 24), [f"Day {d + 1}" for d in range(n_days)], fontsize=8)
    ax.set_ylabel("kWh per hour", fontsize=9)
    ax.set_title(f"First test week of a new client (30 days of history), ε={key[1]:g}", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
