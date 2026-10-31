# Privacy-Preserving Federated Day-Ahead Load Forecasting, with a Battery-Scheduling Decision Layer

A small, reproducible study of **federated learning (FedAvg) with differential privacy (DP-SGD)** for **probabilistic day-ahead** electricity demand forecasting across many consumers, and of what the forecasts are worth when they drive a **decision**: scheduling a home battery under a time-of-use tariff.

Each consumer is a federated client. Raw consumption data never leaves the client; clients only share model weights, which can be trained under a formal (ε, δ)-DP guarantee.

The study asks four questions:

1. **Cold start.** Can a *new* member of an energy community, with only 30 days of history, get a good day-ahead forecast by learning with others, without seeing their data?
2. **Privacy cost.** How much accuracy does a formal guarantee cost, from ε = 1 to ε = 8?
3. **Uncertainty.** Are the predicted 10–90% intervals calibrated?
4. **Decision value.** How much of the savings a perfect forecast would bring does each forecast actually capture when it is used to schedule a battery?

## Setup

| Item | Choice |
|---|---|
| Data | [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014): 370 Portuguese consumers, 15-min readings, resampled to hourly kWh, 2013–2014 |
| Clients | 20 consumers sampled at random from those active over the whole period; 50% are "new members" whose training inputs and targets all fall within their last 30 days (23 training days each) |
| Task | At midnight, forecast the 24 hourly loads of the next day as three quantiles (10%, 50%, 90%), from the past 7 days of hourly readings and calendar features of the target day |
| Split | Chronological: last 20% of days for testing (146 days) |
| Model | MLP (172 → 64 → 64 → 72), trained with the pinball (quantile) loss; kept small on purpose, since DP-SGD adds noise to every parameter's gradient |
| Seeds | 2 training seeds (initialisation, shuffling, DP noise, missing-data mask); same 20 clients; mean and min–max reported |

| Setting | What is shared | Notes |
|---|---|---|
| Naive (yesterday / last week) | nothing | forecast = the profile of 1 or 7 days earlier; standard references in load forecasting |
| Local only | nothing | each client learns alone |
| Centralized | all raw data | upper bound, not privacy-preserving |
| FedAvg | model weights | data stays local; same optimiser (SGD, momentum, gradient clipping) as the DP version |
| FedAvg + DP | model weights trained with DP-SGD (Opacus, RDP accountant) | record-level (ε, δ)-DP per client: one record = one training day; δ = 1/n |
| + pers. | nothing more | each client fine-tunes the global model on its own data; the personalised model never leaves the client |

**Metrics.** NMAE of the median forecast and pinball loss, both in % of the client's mean load; coverage of the 10–90% interval (target 80%).

**Decision layer.** Each client has a battery (capacity 8 h and power 2× its mean hourly load, 95% charge and discharge efficiency) and a time-of-use tariff (0.10 €/kWh 00–07 h, 0.32 €/kWh 18–22 h, 0.20 €/kWh otherwise). Every day, a linear programme plans charging and discharging on the median forecast (no export, state of charge back to 50% at midnight). The plan is applied to the actual load; energy discharged beyond actual demand is lost. The score is the share of the perfect-forecast savings that the plan captures.

## Run

```bash
pip install -r requirements.txt

# with the real data: download and unzip the UCI archive so LD2011_2014.txt sits here
python fl_energy.py

# without data (synthetic consumers with residential/office/industrial profiles)
python fl_energy.py --synthetic
```

It takes about 10–20 minutes on a laptop CPU. Outputs go to `results/`: `results.csv` (mean, min, max over seeds), `results_per_seed.csv`, `results.png` (accuracy), `battery.png` (decision value), `forecast_example.png` (one new client's first test week).

## Results

<!-- RESULTS: filled from the UCI run -->

## Limitations

- **Privacy unit.** DP protects one training day. Consecutive examples share input days, so the guarantee for a single hourly reading is weaker (group privacy). Client-level DP and secure aggregation are not implemented.
- **Scale.** Two seeds, 20 clients, light hyperparameter tuning on synthetic data only, no weather covariates.
- **Missing data.** Missing inputs are imputed with the client mean; the models are not trained to handle them.
- **Decision layer.** One tariff and one battery size per client; the schedule uses only the median forecast.

## Next steps

- Weather covariates and probabilistic calibration (e.g. conformal adjustment of the intervals).
- Decision-aware scheduling that uses the quantiles (risk-averse or stochastic optimisation), and decision-focused training.
- Coordination: a shared community battery or flexibility requests, where clients with competing objectives must agree on a schedule (distributed optimisation, game-theoretic mechanisms).
- Client-level DP, secure aggregation, and robustness to corrupted or adversarial clients.

## Author

Sungyeon Hong · [GitHub](https://github.com/syHong23) · [LinkedIn](https://www.linkedin.com/in/sungyeon-hong-809a59134/)
