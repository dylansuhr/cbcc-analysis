import pandas as pd

# load the raw csv
df = pd.read_csv('data/booking-report-23-24-25.csv')

# print row count and column count
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# strip dollar signs and commas on money columns and convert to numeric
# regex r'[\$,]' matches any '$' or ',' character
# .astype(float) turns the cleaned strings into actual numbers
for col in ['Total', 'Subtotal', 'Total Paid', 'Net Revenue Collected']:
    if col in df.columns:
        df[col + '_num'] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

# list out all the original columns
# enumerate(..., 1) starts the counter at 1 instead of 0
# [:96] limits to the original columns before we added the _num ones
print("\n--- COLUMNS ---")
for i, col in enumerate(df.columns[:96], 1):
    print(f"{i:3}. {col}")

# peek at one full row to see what the data actually looks like
# iloc[0] grabs the first row by position, .to_string() prints it vertically
print("\n--- SAMPLE ROW ---")
print(df.iloc[0].to_string())

# show which columns have missing values and how many
print("\n--- MISSING VALUES ---")
# count nulls per column, then filter to only columns that have any
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
# print each one with its count and percentage of total rows
for col, count in missing.items():
    pct = count / len(df) * 100
    print(f"{col}: {count} ({pct:.0f}%)")

# stats on party size and revenue fields
# .describe() gives count, mean, std, min, 25%, 50%, 75%, max
print("\n--- NUMERIC SUMMARY ---")
print(df[['# of Pax', 'Total_num', 'Subtotal_num', 'Net Revenue Collected_num']].describe().round(2).to_string())

# what items people are booking
# value_counts() tallies each unique value and sorts most common first
print("\n--- ITEM ---")
print(df['Item'].value_counts().to_string())

# where the bookings are coming from
print("\n--- BOOKING CHANNEL (Total Sheet) ---")
print(df['Total Sheet'].value_counts().to_string())

# days of the week values and counts
print("\n--- DAY OF WEEK ---")
print(df['Availability Day'].value_counts().to_string())

# how many bookings got cancelled
print("\n--- CANCELLED ---")
print(df['Cancelled?'].value_counts().to_string())

# paid vs unpaid vs partial
print("\n--- PAID STATUS ---")
print(df['Paid Status'].value_counts().to_string())

# big picture numbers
print("\n" + "="*60)
print(" KEY TAKEAWAYS")
print("="*60)

total_rev = df['Total_num'].sum()
avg_rev = df['Total_num'].mean()
# count rows where cancelled equals 'Cancelled', divide by total rows for the rate
cancel_rate = (df['Cancelled?'] == 'Cancelled').sum() / len(df) * 100
# .index[0] grabs the label of the most frequent value
top_item = df['Item'].value_counts().index[0]
top_channel = df['Total Sheet'].value_counts().index[0]
busiest_day = df['Availability Day'].value_counts().index[0]

# f-string formatting: :,.0f adds commas and rounds to whole dollars
print(f"- Total revenue: ${total_rev:,.0f}")
print(f"- Avg booking: ${avg_rev:,.0f}")
# :.1f rounds to one decimal place
print(f"- Cancellation rate: {cancel_rate:.1f}%")
print(f"- Most popular: {top_item}")
print(f"- Top channel: {top_channel}")
print(f"- Busiest day: {busiest_day}")
# .median() gives the middle value, better than mean for skewed party sizes
print(f"- Median party size: {df['# of Pax'].median():.0f} guests")
