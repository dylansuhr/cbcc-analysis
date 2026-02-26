"""Basic EDA."""

import pandas as pd

df = pd.read_csv('data/booking-report-23-24-25.csv')

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# Parse currency columns
for col in ['Total', 'Subtotal', 'Total Paid', 'Net Revenue Collected']:
    if col in df.columns:
        df[col + '_num'] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

# Columns
print("\n--- COLUMNS ---")
for i, col in enumerate(df.columns[:96], 1):  # original columns only
    print(f"{i:3}. {col}")

# First row vertical
print("\n--- SAMPLE ROW ---")
print(df.iloc[0].to_string())

# Missing - only show columns with missing data
print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, count in missing.items():
    pct = count / len(df) * 100
    print(f"{col}: {count} ({pct:.0f}%)")

# Numeric summary
print("\n--- NUMERIC SUMMARY ---")
print(df[['# of Pax', 'Total_num', 'Subtotal_num', 'Net Revenue Collected_num']].describe().round(2).to_string())

# Key categoricals
print("\n--- ITEM ---")
print(df['Item'].value_counts().to_string())

print("\n--- BOOKING CHANNEL (Total Sheet) ---")
print(df['Total Sheet'].value_counts().to_string())

print("\n--- DAY OF WEEK ---")
print(df['Availability Day'].value_counts().to_string())

print("\n--- CANCELLED ---")
print(df['Cancelled?'].value_counts().to_string())

print("\n--- PAID STATUS ---")
print(df['Paid Status'].value_counts().to_string())

# Conclusions
print("\n" + "="*60)
print(" KEY TAKEAWAYS")
print("="*60)

total_rev = df['Total_num'].sum()
avg_rev = df['Total_num'].mean()
cancel_rate = (df['Cancelled?'] == 'Cancelled').sum() / len(df) * 100
top_item = df['Item'].value_counts().index[0]
top_channel = df['Total Sheet'].value_counts().index[0]
busiest_day = df['Availability Day'].value_counts().index[0]

print(f"- Total revenue: ${total_rev:,.0f}")
print(f"- Avg booking: ${avg_rev:,.0f}")
print(f"- Cancellation rate: {cancel_rate:.1f}%")
print(f"- Most popular: {top_item}")
print(f"- Top channel: {top_channel}")
print(f"- Busiest day: {busiest_day}")
print(f"- Median party size: {df['# of Pax'].median():.0f} guests")
