from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_NUM_PEOPLE, DEFAULT_SEED, DEFAULT_START_DATE, PROFILE_CONFIGS

SCENARIOS = ("simple_rules", "realistic_mixed", "stress_test")
DURATION_YEARS = (1, 2, 3, 5)


@dataclass(frozen=True)
class RealisticPerson:
    person_id: str
    profile_type: str
    monthly_income_ntd: float
    fixed_cost_ratio: float
    food_preference: float
    coffee_preference: float
    commute_intensity: float
    social_intensity: float
    gym_intensity: float
    saving_pressure: float
    spending_volatility: float
    shock_probability: float
    cash_sensitivity: float
    missing_log_probability: float


@dataclass(frozen=True)
class RealisticGenerationResult:
    transactions: pd.DataFrame
    person_profiles: pd.DataFrame
    day_metadata: pd.DataFrame
    generator_metadata: pd.DataFrame


def _scenario_settings(scenario: str) -> dict[str, float]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}. Expected one of {SCENARIOS}")
    settings = {
        "simple_rules": {
            "volatility_scale": 0.65,
            "shock_scale": 0.40,
            "missing_log_scale": 0.35,
            "cash_sensitivity_scale": 0.70,
        },
        "realistic_mixed": {
            "volatility_scale": 1.00,
            "shock_scale": 1.00,
            "missing_log_scale": 1.00,
            "cash_sensitivity_scale": 1.00,
        },
        "stress_test": {
            "volatility_scale": 1.55,
            "shock_scale": 2.10,
            "missing_log_scale": 1.70,
            "cash_sensitivity_scale": 1.25,
        },
    }
    return settings[scenario]


def _end_date_from_years(start_date: str, years: int) -> str:
    if years not in DURATION_YEARS:
        raise ValueError(f"years must be one of {DURATION_YEARS}")
    start = date.fromisoformat(start_date)
    return (date(start.year + years, start.month, start.day) - timedelta(days=1)).isoformat()


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _month_starts(start: date, end: date) -> list[date]:
    return [ts.date() for ts in pd.date_range(start=start, end=end, freq="MS")]


def _inflation_factor(day: date, start: date, annual_rate: float) -> float:
    years_elapsed = (day - start).days / 365.25
    return float((1 + annual_rate) ** years_elapsed)


def _positive_lognormal(rng: np.random.Generator, mean: float, sigma: float, inflation: float = 1.0) -> int:
    sigma = max(0.05, sigma)
    mu = np.log(max(1.0, mean)) - (sigma**2 / 2)
    return max(1, int(round(rng.lognormal(mu, sigma) * inflation)))


def _gamma_amount(rng: np.random.Generator, mean: float, shape: float, inflation: float = 1.0) -> int:
    shape = max(0.5, shape)
    scale = max(1.0, mean) / shape
    return max(1, int(round(rng.gamma(shape, scale) * inflation)))


def _pareto_amount(rng: np.random.Generator, minimum: float, shape: float, inflation: float = 1.0) -> int:
    shape = max(1.2, shape)
    return max(1, int(round((minimum * (1 + rng.pareto(shape))) * inflation)))


def _clip_probability(value: float) -> float:
    return float(min(0.98, max(0.0, value)))


def _sample_people(num_people: int, scenario: str, seed: int) -> list[RealisticPerson]:
    rng = np.random.default_rng(seed)
    settings = _scenario_settings(scenario)
    profiles = list(PROFILE_CONFIGS)
    profile_weights = np.array([profile.weight for profile in profiles], dtype=float)
    profile_weights /= profile_weights.sum()
    people: list[RealisticPerson] = []

    for person_index in range(1, num_people + 1):
        profile = rng.choice(profiles, p=profile_weights)
        income_multiplier = rng.lognormal(np.log(profile.income_multiplier), 0.12)
        volatility = profile.variance_multiplier * settings["volatility_scale"] * rng.uniform(0.75, 1.35)
        social = profile.weekend_social_probability * rng.uniform(0.75, 1.30)
        coffee = profile.coffee_probability * rng.uniform(0.75, 1.25)
        missing_base = rng.uniform(0.005, 0.025) * settings["missing_log_scale"]
        if profile.profile_type == "irregular_high_variance_spender":
            missing_base *= 1.35
        people.append(
            RealisticPerson(
                person_id=f"P{person_index:03d}",
                profile_type=profile.profile_type,
                monthly_income_ntd=float(rng.normal(16500, 2400) * income_multiplier),
                fixed_cost_ratio=float(rng.uniform(0.18, 0.38)),
                food_preference=float(profile.daily_meal_multiplier * rng.uniform(0.85, 1.20)),
                coffee_preference=float(_clip_probability(coffee)),
                commute_intensity=float(profile.transport_multiplier * rng.uniform(0.80, 1.30)),
                social_intensity=float(_clip_probability(social)),
                gym_intensity=float(_clip_probability(profile.gym_probability * rng.uniform(0.70, 1.60))),
                saving_pressure=float(rng.uniform(0.08, 0.38)),
                spending_volatility=float(volatility),
                shock_probability=float(_clip_probability(profile.variance_multiplier * 0.010 * settings["shock_scale"])),
                cash_sensitivity=float(rng.uniform(0.20, 0.65) * settings["cash_sensitivity_scale"]),
                missing_log_probability=float(_clip_probability(missing_base)),
            )
        )
    return people


def _add_record(
    records: list[dict],
    person: RealisticPerson,
    day: date,
    category: str,
    transaction_type: str,
    amount_ntd: int,
    description: str,
    rng: np.random.Generator,
    income_source: str | None = None,
    payment_method: str | None = None,
) -> None:
    records.append(
        {
            "person_id": person.person_id,
            "profile_type": person.profile_type,
            "date": day.isoformat(),
            "category": category,
            "transaction_type": transaction_type,
            "amount_ntd": int(amount_ntd),
            "day_of_week": day.weekday(),
            "is_weekend": int(day.weekday() >= 5),
            "month": day.month,
            "payday_distance": min(abs(day.day - 1), abs(day.day - 5), abs(day.day - 20), abs(day.day - 25)),
            "income_source": income_source,
            "payment_method": payment_method or ("cash" if rng.random() < 0.58 else "card"),
            "description": description,
        }
    )


def _discretionary_pressure(balance: float, monthly_income: float, saving_pressure: float, cash_sensitivity: float) -> float:
    target_balance = monthly_income * saving_pressure
    if balance >= target_balance:
        return 1.0
    shortfall = (target_balance - balance) / max(monthly_income, 1)
    return float(max(0.25, 1.0 - shortfall * cash_sensitivity * 2.0))


def generate_realistic_dataset(
    num_people: int = DEFAULT_NUM_PEOPLE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    years: int = 1,
    scenario: str = "realistic_mixed",
    annual_inflation_rate: float = 0.025,
    seed: int = DEFAULT_SEED,
) -> RealisticGenerationResult:
    """Generate a more realistic synthetic finance dataset without replacing the baseline generator."""
    if end_date is None:
        end_date = _end_date_from_years(start_date, years)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    rng = np.random.default_rng(seed)
    people = _sample_people(num_people, scenario, seed)
    months = _month_starts(start, end)
    records: list[dict] = []
    day_metadata: list[dict] = []

    for person in people:
        balance = float(person.monthly_income_ntd * rng.uniform(0.15, 0.55))
        recovery_days = 0
        exam_months = {1, 6, 12}

        for day in _date_range(start, end):
            inflation = _inflation_factor(day, start, annual_inflation_rate)
            missing_day = bool(rng.random() < person.missing_log_probability)
            low_cash = balance < person.monthly_income_ntd * person.saving_pressure * 0.65
            exam_period = day.month in exam_months and day.day >= 18
            weekend = day.weekday() >= 5
            social_state = weekend and rng.random() < person.social_intensity * (0.65 if low_cash else 1.0)
            shock_day = bool(rng.random() < person.shock_probability)
            day_metadata.append(
                {
                    "person_id": person.person_id,
                    "profile_type": person.profile_type,
                    "date": day.isoformat(),
                    "missing_log_day": int(missing_day),
                    "low_cash_mode": int(low_cash),
                    "exam_period": int(exam_period),
                    "social_state": int(social_state),
                    "shock_day": int(shock_day),
                    "balance_start_ntd": round(balance, 2),
                }
            )
            if missing_day:
                continue

            day_total = 0
            if day.day in (1, 20):
                source = "family_support" if day.day == 1 else "scholarship"
                amount = _gamma_amount(rng, person.monthly_income_ntd * (0.55 if day.day == 1 else 0.35), 9, inflation)
                _add_record(records, person, day, "income", "income", amount, f"{source} income", rng, source, "bank_transfer")
                balance += amount
            if person.profile_type == "part_time_worker" and day.day in (5, 25):
                amount = _gamma_amount(rng, person.monthly_income_ntd * 0.20, 5, inflation)
                _add_record(records, person, day, "income", "income", amount, "Part-time work income", rng, "part_time_work", "bank_transfer")
                balance += amount

            if day.day == 1:
                fixed_total = person.monthly_income_ntd * person.fixed_cost_ratio
                bills = [
                    ("rent_or_dorm", fixed_total * 0.50, "Monthly rent or dorm fee"),
                    ("transport", 620 * person.commute_intensity, "Transport pass or commute top-up"),
                    ("subscription", 380, "Monthly subscriptions"),
                    ("groceries", fixed_total * 0.18 * person.food_preference, "Monthly groceries"),
                ]
                for category, mean, description in bills:
                    amount = _gamma_amount(rng, mean, 12, inflation)
                    _add_record(records, person, day, category, "expense", amount, description, rng)
                    balance -= amount
                    day_total += amount
            if day.month in (2, 9) and day.day == 10:
                amount = _gamma_amount(rng, 2600, 4, inflation)
                _add_record(records, person, day, "school_supplies", "expense", amount, "Semester materials", rng)
                balance -= amount
                day_total += amount

            pressure = _discretionary_pressure(balance, person.monthly_income_ntd, person.saving_pressure, person.cash_sensitivity)
            if recovery_days > 0:
                pressure *= 0.62
                recovery_days -= 1

            lunch_probability = _clip_probability((0.86 if not weekend else 0.72) * pressure)
            dinner_probability = _clip_probability((0.84 if not weekend else 0.90) * pressure)
            coffee_probability = _clip_probability(person.coffee_preference * (0.70 if low_cash else 1.0) * (0.75 if exam_period else 1.0))
            gym_probability = _clip_probability(person.gym_intensity * (0.65 if low_cash else 1.0))

            if rng.random() < lunch_probability:
                amount = _positive_lognormal(rng, 95 * person.food_preference, 0.22 * person.spending_volatility, inflation)
                _add_record(records, person, day, "meal", "expense", amount, "Lunch meal", rng)
                balance -= amount
                day_total += amount
            if rng.random() < dinner_probability:
                meal_multiplier = person.food_preference * (1.18 if weekend else 1.0)
                amount = _positive_lognormal(rng, 135 * meal_multiplier, 0.28 * person.spending_volatility, inflation)
                _add_record(records, person, day, "meal", "expense", amount, "Dinner meal", rng)
                balance -= amount
                day_total += amount
            if rng.random() < coffee_probability:
                amount = _positive_lognormal(rng, 58, 0.20 * person.spending_volatility, inflation)
                _add_record(records, person, day, "coffee", "expense", amount, "Coffee purchase", rng)
                balance -= amount
                day_total += amount
            if day.weekday() == 5 and rng.random() < 0.82:
                amount = _gamma_amount(rng, 70, 10, inflation)
                _add_record(records, person, day, "laundry", "expense", amount, "Weekly laundry", rng)
                balance -= amount
                day_total += amount
            if rng.random() < gym_probability / 7:
                amount = _positive_lognormal(rng, 180, 0.45 * person.spending_volatility, inflation)
                _add_record(records, person, day, "gym", "expense", amount, "Gym or supplement purchase", rng)
                balance -= amount
                day_total += amount
            if social_state:
                amount = _positive_lognormal(rng, 520 * person.food_preference, 0.55 * person.spending_volatility, inflation)
                _add_record(records, person, day, "social_food", "expense", amount, "Weekend meal or social event", rng)
                balance -= amount
                day_total += amount
                if rng.random() < 0.55:
                    transport_amount = _gamma_amount(rng, 160 * person.commute_intensity, 4, inflation)
                    _add_record(records, person, day, "transport", "expense", transport_amount, "Social commute", rng)
                    balance -= transport_amount
                    day_total += transport_amount
            if shock_day:
                amount = _pareto_amount(rng, 850, 1.85 / max(person.spending_volatility, 0.7), inflation)
                _add_record(records, person, day, "irregular_purchase", "expense", amount, "Irregular high-value purchase", rng)
                balance -= amount
                day_total += amount

            if day_total > person.monthly_income_ntd * 0.10:
                recovery_days = int(rng.integers(1, 4))

    transactions = pd.DataFrame.from_records(records)
    if not transactions.empty:
        transactions = transactions.sort_values(["person_id", "date", "transaction_type", "category"]).reset_index(drop=True)
    person_profiles = pd.DataFrame([asdict(person) for person in people])
    day_metadata_df = pd.DataFrame.from_records(day_metadata)
    generator_metadata = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "num_people": num_people,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "years": years,
                "annual_inflation_rate": annual_inflation_rate,
                "seed": seed,
                "transaction_rows": len(transactions),
                "missing_log_days": int(day_metadata_df["missing_log_day"].sum()),
            }
        ]
    )
    return RealisticGenerationResult(transactions, person_profiles, day_metadata_df, generator_metadata)


def save_realistic_dataset(result: RealisticGenerationResult, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "transactions": output / "realistic_expense_transactions.csv",
        "person_profiles": output / "realistic_person_profiles.csv",
        "day_metadata": output / "realistic_day_metadata.csv",
        "generator_metadata": output / "realistic_generator_metadata.csv",
    }
    result.transactions.to_csv(paths["transactions"], index=False)
    result.person_profiles.to_csv(paths["person_profiles"], index=False)
    result.day_metadata.to_csv(paths["day_metadata"], index=False)
    result.generator_metadata.to_csv(paths["generator_metadata"], index=False)
    return paths
