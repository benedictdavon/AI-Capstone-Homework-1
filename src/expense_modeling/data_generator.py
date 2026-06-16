from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from random import Random

import pandas as pd

from .config import DEFAULT_END_DATE, DEFAULT_NUM_PEOPLE, DEFAULT_SEED, DEFAULT_START_DATE, PROFILE_CONFIGS


def _biased_amount(rng: Random, low: int, high: int, multiplier: float = 1.0, variance: float = 1.0) -> int:
    values = list(range(low, high + 1))
    weights = [2 if value % 10 in (0, 5) else 1 for value in values]
    amount = rng.choices(values, weights=weights, k=1)[0] * multiplier
    jitter = rng.uniform(max(0.55, 1 - 0.18 * variance), 1 + 0.18 * variance)
    return max(1, int(round(amount * jitter)))


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _month_starts(start: date, end: date) -> list[date]:
    months = pd.date_range(start=start, end=end, freq="MS")
    return [ts.date() for ts in months]


def _add_record(
    records: list[dict],
    *,
    person_id: str,
    profile_type: str,
    day: date,
    category: str,
    transaction_type: str,
    amount_ntd: int,
    description: str,
    rng: Random,
    income_source: str | None = None,
    payment_method: str | None = None,
) -> None:
    records.append(
        {
            "person_id": person_id,
            "profile_type": profile_type,
            "date": day.isoformat(),
            "category": category,
            "transaction_type": transaction_type,
            "amount_ntd": int(amount_ntd),
            "day_of_week": day.weekday(),
            "is_weekend": int(day.weekday() >= 5),
            "month": day.month,
            "payday_distance": min(abs(day.day - 1), abs(day.day - 5), abs(day.day - 20), abs(day.day - 25)),
            "income_source": income_source,
            "payment_method": payment_method or ("cash" if rng.random() < 0.65 else "card"),
            "description": description,
        }
    )


def generate_transactions(
    num_people: int = DEFAULT_NUM_PEOPLE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate synthetic transaction-level personal finance data."""
    rng = Random(seed)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    profiles = list(PROFILE_CONFIGS)
    profile_weights = [profile.weight for profile in profiles]
    months = _month_starts(start, end)
    records: list[dict] = []

    for person_idx in range(1, num_people + 1):
        profile = rng.choices(profiles, weights=profile_weights, k=1)[0]
        person_id = f"P{person_idx:03d}"
        base_income = _biased_amount(rng, 7800, 9800, profile.income_multiplier, profile.variance_multiplier)
        scholarship = _biased_amount(rng, 4500, 6500, profile.income_multiplier, 0.7)

        for month_start in months:
            parent_day = date(month_start.year, month_start.month, rng.randint(1, 3))
            _add_record(
                records,
                person_id=person_id,
                profile_type=profile.profile_type,
                day=parent_day,
                category="income",
                transaction_type="income",
                amount_ntd=base_income,
                income_source="family_support",
                description="Monthly family support",
                rng=rng,
                payment_method="bank_transfer",
            )
            scholarship_day = date(month_start.year, month_start.month, 20)
            _add_record(
                records,
                person_id=person_id,
                profile_type=profile.profile_type,
                day=scholarship_day,
                category="income",
                transaction_type="income",
                amount_ntd=scholarship,
                income_source="scholarship",
                description="Monthly scholarship income",
                rng=rng,
                payment_method="bank_transfer",
            )
            if rng.random() < profile.part_time_probability:
                work_day = date(month_start.year, month_start.month, min(rng.randint(24, 28), 28))
                _add_record(
                    records,
                    person_id=person_id,
                    profile_type=profile.profile_type,
                    day=work_day,
                    category="income",
                    transaction_type="income",
                    amount_ntd=_biased_amount(rng, 2500, 6200, profile.income_multiplier, profile.variance_multiplier),
                    income_source="part_time_work",
                    description="Part-time work income",
                    rng=rng,
                    payment_method="bank_transfer",
                )

            fixed_expenses = [
                ("transport", 220, 520, profile.transport_multiplier, "Monthly transport pass and commuting"),
                ("subscription", 99, 699, 1.0, "Streaming and productivity subscriptions"),
                ("groceries", 550, 1200, profile.daily_meal_multiplier, "Monthly groceries and household items"),
            ]
            for category, low, high, multiplier, description in fixed_expenses:
                _add_record(
                    records,
                    person_id=person_id,
                    profile_type=profile.profile_type,
                    day=month_start,
                    category=category,
                    transaction_type="expense",
                    amount_ntd=_biased_amount(rng, low, high, multiplier, profile.variance_multiplier),
                    description=description,
                    rng=rng,
                )

            if rng.random() < profile.gym_probability:
                _add_record(
                    records,
                    person_id=person_id,
                    profile_type=profile.profile_type,
                    day=month_start + timedelta(days=min(rng.randint(0, 6), 27)),
                    category="gym",
                    transaction_type="expense",
                    amount_ntd=_biased_amount(rng, 600, 1800, 1.0, profile.variance_multiplier),
                    description="Gym membership or supplement purchase",
                    rng=rng,
                )

        for day in _date_range(start, end):
            _add_record(
                records,
                person_id=person_id,
                profile_type=profile.profile_type,
                day=day,
                category="meal",
                transaction_type="expense",
                amount_ntd=_biased_amount(rng, 70, 120, profile.daily_meal_multiplier, profile.variance_multiplier),
                description="Lunch meal",
                rng=rng,
            )
            dinner_multiplier = profile.daily_meal_multiplier * (1.15 if day.weekday() >= 5 else 1.0)
            _add_record(
                records,
                person_id=person_id,
                profile_type=profile.profile_type,
                day=day,
                category="meal",
                transaction_type="expense",
                amount_ntd=_biased_amount(rng, 85, 180, dinner_multiplier, profile.variance_multiplier),
                description="Dinner meal",
                rng=rng,
            )
            if rng.random() < profile.coffee_probability:
                _add_record(
                    records,
                    person_id=person_id,
                    profile_type=profile.profile_type,
                    day=day,
                    category="coffee",
                    transaction_type="expense",
                    amount_ntd=_biased_amount(rng, 40, 75, 1.0, profile.variance_multiplier),
                    description="Coffee purchase",
                    rng=rng,
                )
            if day.weekday() == 5:
                _add_record(
                    records,
                    person_id=person_id,
                    profile_type=profile.profile_type,
                    day=day,
                    category="laundry",
                    transaction_type="expense",
                    amount_ntd=_biased_amount(rng, 50, 80, 1.0, 0.5),
                    description="Weekly laundry",
                    rng=rng,
                )
            if day.weekday() >= 5 and rng.random() < profile.weekend_social_probability:
                _add_record(
                    records,
                    person_id=person_id,
                    profile_type=profile.profile_type,
                    day=day,
                    category="social_food",
                    transaction_type="expense",
                    amount_ntd=_biased_amount(rng, 220, 900, profile.daily_meal_multiplier, profile.variance_multiplier),
                    description="Weekend meal or social event",
                    rng=rng,
                )
            if profile.profile_type == "irregular_high_variance_spender" and rng.random() < 0.025:
                _add_record(
                    records,
                    person_id=person_id,
                    profile_type=profile.profile_type,
                    day=day,
                    category="irregular_purchase",
                    transaction_type="expense",
                    amount_ntd=_biased_amount(rng, 800, 4500, 1.0, 2.3),
                    description="Irregular high-value purchase",
                    rng=rng,
                )

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["person_id", "date", "transaction_type", "category"]).reset_index(drop=True)


def save_transactions(df: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def profile_summary() -> list[dict]:
    return [asdict(profile) for profile in PROFILE_CONFIGS]
