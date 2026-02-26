"""
Data loading and cleaning module for CBCC booking data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def parse_currency(value):
    """Parse currency string to float."""
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace('$', '').replace(',', ''))


def parse_time_to_hours(start_time: str, end_time: str) -> float:
    """Calculate duration in hours between start and end time strings."""
    try:
        start = pd.to_datetime(start_time, format='%I:%M %p')
        end = pd.to_datetime(end_time, format='%I:%M %p')
        duration = (end - start).total_seconds() / 3600
        if duration < 0:
            duration += 24  # Handle overnight trips
        return duration
    except:
        return np.nan


def extract_charter_category(item_name: str) -> dict:
    """Extract vessel and charter type from Item name."""
    if pd.isna(item_name):
        return {'vessel': 'Unknown', 'charter_type': 'Unknown'}

    item_upper = item_name.upper()

    # Determine vessel
    if 'MARIE L' in item_upper or 'ML' in item_upper:
        vessel = 'Marie L'
    elif 'MONHEGAN' in item_upper:
        vessel = 'Monhegan'
    elif 'ELIZABETH' in item_upper:
        vessel = 'Elizabeth Grace'
    elif '6-PACK' in item_upper:
        vessel = '6-Pack'
    elif 'CASCO BAY' in item_upper:
        vessel = 'Casco Bay'
    else:
        vessel = 'Other'

    # Determine charter type
    item_lower = item_name.lower()
    if 'sunset' in item_lower:
        charter_type = 'Sunset'
    elif 'brunch' in item_lower:
        charter_type = 'Brunch'
    elif 'dinner' in item_lower:
        charter_type = 'Dinner'
    elif 'sightseeing' in item_lower:
        charter_type = 'Sightseeing'
    elif 'transportation' in item_lower:
        charter_type = 'Transportation'
    elif 'pick 3' in item_lower:
        charter_type = 'Pick 3'
    else:
        charter_type = 'Other'

    return {'vessel': vessel, 'charter_type': charter_type}


def load_booking_data(filepath: str | Path = 'data/booking-report-23-24-25.csv') -> pd.DataFrame:
    """
    Load and clean CBCC booking data.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned booking data with derived features.
    """
    # Load data, skipping the title row
    df = pd.read_csv(filepath, skiprows=1)

    # Parse currency columns
    currency_cols = [
        'Total', 'Subtotal', 'Total Tax', 'Total Paid',
        'Dashboard Tax Rate (0%)', 'Food and Beverage Tax (8%)',
        'All-Inclusive Flat Tax ($3.20)', 'MH Dinner Guests 1 & 2 ($19.20)',
        'MH Dinner Guests 3 - 6 ($14.40)', 'ML Mimosa Brunch ($6.00)',
        'Staff Wellness (5%)', 'Net Revenue Collected', 'Processing Fees'
    ]

    for col in currency_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_currency)

    # Parse date columns
    date_cols = ['Created At Date', 'Start Date', 'End Date', 'Cancelled At Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')

    # Filter out cancelled bookings
    df = df[df['Cancelled?'] != 'Cancelled'].copy()

    # Filter out test bookings and outliers
    df = df[df['Contact'].str.lower() != 'test'].copy()
    df = df[df['# of Pax'] <= 100].copy()  # Remove aggregate rows
    df = df[df['Total'] > 0].copy()  # Remove zero-revenue bookings

    # Derive features
    # Lead time (days between booking and charter)
    df['lead_time_days'] = (df['Start Date'] - df['Created At Date']).dt.days

    # Charter duration in hours
    df['charter_duration_hours'] = df.apply(
        lambda row: parse_time_to_hours(row['Start Time'], row['End Time']),
        axis=1
    )

    # Revenue per guest
    df['revenue_per_guest'] = df['Total'] / df['# of Pax']

    # Temporal features from Start Date
    df['month'] = df['Start Date'].dt.month
    df['month_name'] = df['Start Date'].dt.month_name()
    df['week'] = df['Start Date'].dt.isocalendar().week
    df['day_of_week'] = df['Start Date'].dt.day_name()
    df['year'] = df['Start Date'].dt.year

    # Extract charter category
    category_data = df['Item'].apply(extract_charter_category)
    df['vessel'] = category_data.apply(lambda x: x['vessel'])
    df['charter_type'] = category_data.apply(lambda x: x['charter_type'])

    # Booking channel (from Total Sheet column)
    df['booking_channel'] = df['Total Sheet'].fillna('Unknown')

    return df


if __name__ == '__main__':
    # Test loading
    df = load_booking_data()
    print(f"Loaded {len(df)} bookings")
    print(f"Date range: {df['Start Date'].min()} to {df['Start Date'].max()}")
    print(f"Columns: {list(df.columns)}")
