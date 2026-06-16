# Data Generation Scripts

This folder has two generators

`generate_data.py` is the original baseline generator

`generate_realistic_data.py` is the newer realistic generator for benchmark-style experiments

The baseline generator is kept so model results can be compared across data versions

## Baseline Generator

Run:

```bash
python scripts/generate_data.py --num-people 50 --seed 42
```

Output:

```text
data/raw/synthetic_expense_transactions.csv
```

The baseline generator uses simpler profile multipliers and regular daily transactions

It is useful as a sanity check because the patterns are easier to learn

## Realistic Generator

Run:

```bash
python scripts/generate_realistic_data.py \
  --num-people 50 \
  --years 1 \
  --scenario realistic_mixed \
  --annual-inflation-rate 0.025 \
  --seed 42
```

Generated files:

```text
data/raw/realistic_expense_transactions.csv
data/raw/realistic_person_profiles.csv
data/raw/realistic_day_metadata.csv
data/raw/realistic_generator_metadata.csv
```

Use the realistic transactions with the existing model scripts:

```bash
python scripts/run_supervised.py --transactions data/raw/realistic_expense_transactions.csv
python scripts/run_clustering.py --transactions data/raw/realistic_expense_transactions.csv
```

## Dataset Comparison

Run the full comparison matrix:

```bash
python scripts/run_dataset_comparison.py --num-people 50 --seed 42 --annual-inflation-rate 0.025
```

Output:

```text
results/metrics/dataset_comparison.csv
```

The comparison currently includes:

- baseline synthetic
- `simple_rules` for 1 year
- `realistic_mixed` for 1, 2, 3, and 5 years
- `stress_test` for 1 year

## Duration Options

The realistic generator supports longer spending histories:

```bash
python scripts/generate_realistic_data.py --years 2
python scripts/generate_realistic_data.py --years 3
python scripts/generate_realistic_data.py --years 5
```

Supported values:

- `1`
- `2`
- `3`
- `5`

An explicit `--end-date` can be used when a custom range is needed

## Scenarios

`simple_rules`

- Lower noise
- Fewer missing log days
- Fewer large shocks
- Good for checking whether the model pipeline works

`realistic_mixed`

- Default scenario
- Uses payday effects, missing logs, heavy-tailed irregular purchases, inflation, and person-level habits
- Best default for model comparison

`stress_test`

- Higher variance
- More missing log days
- More large irregular purchases
- Useful for testing whether features are robust

## Person Model

Each person gets stable traits:

- profile type
- monthly income
- fixed cost ratio
- food preference
- coffee preference
- commute intensity
- social intensity
- gym intensity
- saving pressure
- spending volatility
- shock probability
- cash sensitivity
- missing log probability

People with the same profile type are similar, but not identical

## Spending Model

The realistic generator uses an agent-style daily simulation

Each person has a simple balance that changes with income and expenses

Daily spending depends on:

- day of week
- weekend status
- payday timing
- current balance
- saving pressure
- previous high-spend days
- exam periods
- social weekend state
- random shocks

This makes spending less deterministic than the baseline generator

## Amount Distributions

The generator uses different distributions by spending type:

| Category | Distribution idea |
| --- | --- |
| Meals and coffee | Lognormal |
| Groceries and bills | Gamma |
| Transport | Gamma plus commute intensity |
| Social food | Lognormal with weekend weighting |
| Irregular purchases | Pareto-style heavy tail |
| Income | Gamma around monthly schedule |

The goal is to create many small purchases and a few larger spikes

## Inflation

Inflation is applied through an annual growth factor:

```text
amount_today = sampled_amount * (1 + annual_inflation_rate) ^ years_elapsed
```

Default:

```text
annual_inflation_rate = 0.025
```

For a five-year dataset, later transactions should be slightly larger on average

## Missing Log Days

Some people forget to record spending on some days

When a day is missed:

- no transaction rows are emitted for that person-day
- the missing day is recorded in `realistic_day_metadata.csv`
- the model should not treat the day as true zero spending

This is closer to manually tracked spending logs, where missing records are common

## Metadata Files

`realistic_person_profiles.csv`

- one row per person
- includes the latent traits used by the simulator
- useful for analysis, but should not be used as normal model input unless testing oracle features

`realistic_day_metadata.csv`

- one row per person-day
- includes missing log flags, low-cash mode, exam period, social state, shock day, and starting balance

`realistic_generator_metadata.csv`

- one row with scenario, date range, seed, inflation rate, transaction count, and missing log count

## Recommended Comparisons

Run the same model pipeline on:

```text
baseline synthetic data
realistic_mixed 1-year data
realistic_mixed 3-year data
stress_test 1-year data
```

Then compare:

- RMSE
- MAE
- R2
- feature importance
- baseline model rows

This makes it easier to see which features work only on simple rules and which still help under noisier synthetic data
