import pandas as pd
import random
from datetime import datetime, timedelta, date

# Set random seed for reproducibility
random.seed(42)

# Helper function to generate a whole number with bias:
def biased_amount(min_val, max_val):
    """
    Generate a whole number between min_val and max_val (inclusive)
    with bias: numbers ending in 0 or 5 have higher probability.
    """
    values = list(range(min_val, max_val + 1))
    weights = [2 if (val % 10 == 0 or val % 10 == 5) else 1 for val in values]
    return random.choices(values, weights=weights, k=1)[0]

# Simulation period: one year (365 days)
start_date = date(2025, 1, 1)
num_days = 365
end_date = start_date + timedelta(days=num_days - 1)

records = []

# Create a list of month start dates for monthly records
months = pd.date_range(start=start_date, end=end_date, freq='MS')

# --- Revised Monthly Income ---

for m in months:
    m_date = m.date()
    
    # Scholarship: fixed 6000 NTD paid on the 20th of the same month.
    try:
        scholarship_date = date(m_date.year, m_date.month, 20)
    except ValueError:
        # In case the month doesn't have 20 days (shouldn't occur)
        scholarship_date = date(m_date.year, m_date.month, 20)
    records.append({
        'Date': scholarship_date,
        'Category': 'Income',
        'Amount_NTD': 6000,
        'Description': f"Scholarship income for {m_date.strftime('%B %Y')}",
        'Payment_Method': None,
        'Time': None
    })
    
    # Parents: fixed 9000 NTD paid on a random day between 1st and 3rd of the same month.
    parent_pay_day = random.randint(1, 3)
    parents_date = date(m_date.year, m_date.month, parent_pay_day)
    records.append({
        'Date': parents_date,
        'Category': 'Income',
        'Amount_NTD': 9000,
        'Description': f"Parents' income for {m_date.strftime('%B %Y')}",
        'Payment_Method': None,
        'Time': None
    })
    
    # Burger King income: paid on the 5th of the same month.
    bk_amount = biased_amount(1500, 4000)
    bk_pay_date = date(m_date.year, m_date.month, 5)
    records.append({
        'Date': bk_pay_date,
        'Category': 'Income',
        'Amount_NTD': bk_amount,
        'Description': f"Burger King income for {m_date.strftime('%B %Y')}",
        'Payment_Method': None,
        'Time': None
    })
    
    # Office Assistant income: paid on a random day between 25 and 30 of the next month.
    next_month_year = m_date.year
    next_month = m_date.month + 1
    if next_month > 12:
        next_month = 1
        next_month_year += 1
    oa_pay_day = random.randint(25, 30)
    try:
        oa_pay_date = date(next_month_year, next_month, oa_pay_day)
    except ValueError:
        # Adjust if the day is invalid (e.g., February 30)
        oa_pay_date = date(next_month_year, next_month, 28)
    # Only add office assistant income if it falls within the simulation period.
    if start_date <= oa_pay_date <= end_date:
        oa_amount = biased_amount(4500, 5500)
        records.append({
            'Date': oa_pay_date,
            'Category': 'Income',
            'Amount_NTD': oa_amount,
            'Description': f"Office assistant income for {m_date.strftime('%B %Y')}",
            'Payment_Method': None,
            'Time': None
        })
    
    # Fixed monthly expenses (Transport, Entertainment, Groceries)
    # Transport expense: between 200 and 500 NTD
    records.append({
        'Date': m_date,
        'Category': 'Transport',
        'Amount_NTD': biased_amount(200, 500),
        'Description': 'Monthly transport expense',
        'Payment_Method': None,
        'Time': None
    })
    
    # Entertainment subscriptions: fixed monthly amounts
    records.append({
        'Date': m_date,
        'Category': 'Entertainment',
        'Amount_NTD': 99,
        'Description': 'Spotify subscription (student)',
        'Payment_Method': None,
        'Time': None
    })
    records.append({
        'Date': m_date,
        'Category': 'Entertainment',
        'Amount_NTD': 600,
        'Description': 'ChatGPT Plus subscription',
        'Payment_Method': None,
        'Time': None
    })
    records.append({
        'Date': m_date,
        'Category': 'Entertainment',
        'Amount_NTD': 200,
        'Description': 'WuxiaWorld subscription',
        'Payment_Method': None,
        'Time': None
    })
    
    # Groceries (household items): between 500 and 1000 NTD
    records.append({
        'Date': m_date,
        'Category': 'Groceries',
        'Amount_NTD': biased_amount(500, 1000),
        'Description': 'Monthly groceries/household items',
        'Payment_Method': None,
        'Time': None
    })

# Occasional special weekend events: one per semester (2 per year)
def get_random_weekend(start, end):
    """Return a random weekend date (Saturday or Sunday) between start and end."""
    delta = (end - start).days
    while True:
        rand_day = start + timedelta(days=random.randint(0, delta))
        if rand_day.weekday() >= 5:
            return rand_day

first_half = (start_date, date(start_date.year, 6, 30))
second_half = (date(start_date.year, 7, 1), end_date)
special_events = [
    {'Description': 'Korean BBQ with friends', 'Amount_NTD': 700},
    {'Description': 'Haidilao with friends', 'Amount_NTD': 500}
]
special_date1 = get_random_weekend(first_half[0], first_half[1])
special_date2 = get_random_weekend(second_half[0], second_half[1])
special_dates = [special_date1, special_date2]

for i, event in enumerate(special_events):
    records.append({
        'Date': special_dates[i],
        'Category': 'Food & Drink',
        'Amount_NTD': event['Amount_NTD'],
        'Description': event['Description'],
        'Payment_Method': None,
        'Time': 'Evening'
    })

# Gym supplements
protein_months = random.sample(list(months), 2)
for m in protein_months:
    records.append({
        'Date': m.date(),
        'Category': 'Gym',
        'Amount_NTD': 1500,
        'Description': 'Protein powder purchase (2.5kg)',
        'Payment_Method': None,
        'Time': None
    })
creatine_months = random.sample(list(months), 4)
for m in creatine_months:
    records.append({
        'Date': m.date(),
        'Category': 'Gym',
        'Amount_NTD': 600,
        'Description': 'Creatine purchase',
        'Payment_Method': None,
        'Time': None
    })

# Define gym days: roughly 3 days per week
gym_days = set()
for day in range(num_days):
    current_date = start_date + timedelta(days=day)
    if random.random() < 3/7:
        gym_days.add(current_date)

# Simulate daily expenses for meals, coffee, and laundry.
for day in range(num_days):
    current_date = start_date + timedelta(days=day)
    weekday = current_date.weekday()  # Monday=0, Sunday=6

    # Lunch at around 12:00 PM (Meal expense: 80-105 NTD)
    lunch_time = "12:00"
    lunch_amount = biased_amount(80, 105)
    lunch_payment = 'Cash' if random.random() < 0.7 else 'Card'
    records.append({
        'Date': current_date,
        'Category': 'Meal',
        'Amount_NTD': lunch_amount,
        'Description': 'Lunch meal',
        'Payment_Method': lunch_payment,
        'Time': lunch_time
    })

    # Coffee: approximately 3 times per week around 12:20-13:00 PM (40-60 NTD)
    if random.random() < (3/7):
        coffee_time = f"12:{random.randint(20, 59):02d}"
        coffee_amount = biased_amount(40, 60)
        coffee_payment = 'Cash' if random.random() < 0.7 else 'Card'
        records.append({
            'Date': current_date,
            'Category': 'Coffee',
            'Amount_NTD': coffee_amount,
            'Description': 'Coffee purchase',
            'Payment_Method': coffee_payment,
            'Time': coffee_time
        })

    # Dinner: between 5 PM and 8 PM.
    # On gym days, dinner is more expensive (100-150 NTD);
    # on weekends, moderately higher (100-250 NTD);
    # otherwise, normal dinner: 80-110 NTD.
    dinner_hour = random.randint(17, 20)
    dinner_minute = random.randint(0, 59)
    dinner_time = f"{dinner_hour}:{dinner_minute:02d}"
    if current_date in gym_days:
        dinner_amount = biased_amount(100, 150)
    elif weekday >= 5:  # weekend
        dinner_amount = biased_amount(100, 250)
    else:
        dinner_amount = biased_amount(80, 110)
    dinner_payment = 'Cash' if random.random() < 0.7 else 'Card'
    records.append({
        'Date': current_date,
        'Category': 'Meal',
        'Amount_NTD': dinner_amount,
        'Description': 'Dinner meal',
        'Payment_Method': dinner_payment,
        'Time': dinner_time
    })

    # Laundry: 60 NTD per week on Saturday
    if current_date.weekday() == 5:
        records.append({
            'Date': current_date,
            'Category': 'Laundry',
            'Amount_NTD': 60,
            'Description': 'Weekly laundry expense',
            'Payment_Method': None,
            'Time': None
        })

# Create DataFrame and sort by Date and Time
df = pd.DataFrame(records)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by=['Date', 'Time'], inplace=True)

# Save the dataset to a CSV file and preview the first 20 records.
df.to_csv('daily_expenses.csv', index=False)
print("Data generation complete. Preview of the first 20 records:")
print(df.head(20))
