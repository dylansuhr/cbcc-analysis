"""
Exploratory Data Analysis for CBCC booking data.

Generates summary statistics and visualizations for charter booking patterns.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.load_data import load_booking_data

# Configuration
OUTPUT_DIR = Path('output/figures')
REPORT_PATH = Path('output/eda_report.md')
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE = (10, 6)

# Report file handle (set in main)
_report_file = None


def write(text: str = ""):
    """Write to both stdout and report file."""
    print(text)
    if _report_file:
        _report_file.write(text + "\n")


def save_figure(fig, name: str):
    """Save figure to output directory."""
    filepath = OUTPUT_DIR / f'{name}.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    write()
    write(f'<img src="figures/{name}.png" width=800>')
    write()


def section_header(title: str):
    """Print section header."""
    write()
    write(f"## {title}")
    write()


# =============================================================================
# Section A: Data Overview
# =============================================================================
def data_overview(df: pd.DataFrame):
    """Print data overview statistics."""
    section_header("A. Data Overview")

    write(f"**Dataset Shape:** {df.shape[0]} rows x {df.shape[1]} columns")
    write(f"**Date Range:** {df['Start Date'].min().strftime('%Y-%m-%d')} to {df['Start Date'].max().strftime('%Y-%m-%d')}")
    write()

    write("### Missing Values (top 10)")
    write()
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'count': missing, 'percent': missing_pct})
    write(missing_df[missing_df['count'] > 0].head(10).to_markdown())
    write()

    write("### Numeric Summary Statistics")
    write()
    numeric_cols = ['Total', '# of Pax', 'lead_time_days', 'revenue_per_guest', 'charter_duration_hours']
    write(df[numeric_cols].describe().round(2).to_markdown())
    write()

    write("### Bookings by Year")
    write()
    write(df['year'].value_counts().sort_index().to_markdown())
    write()

    write("### Charter Types")
    write()
    write(df['charter_type'].value_counts().to_markdown())
    write()

    write("### Vessels")
    write()
    write(df['vessel'].value_counts().to_markdown())


# =============================================================================
# Section B: Revenue Analysis
# =============================================================================
def revenue_analysis(df: pd.DataFrame):
    """Generate revenue analysis plots."""
    section_header("B. Revenue Analysis")

    # B1: Revenue distribution
    fig, ax = plt.subplots(figsize=FIGSIZE)
    df['Total'].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Total Revenue ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Booking Revenue')
    ax.axvline(df['Total'].median(), color='red', linestyle='--', label=f"Median: ${df['Total'].median():,.0f}")
    ax.axvline(df['Total'].mean(), color='orange', linestyle='--', label=f"Mean: ${df['Total'].mean():,.0f}")
    ax.legend()
    save_figure(fig, 'b1_revenue_distribution')

    # B2: Revenue by charter type
    fig, ax = plt.subplots(figsize=FIGSIZE)
    revenue_by_type = df.groupby('charter_type')['Total'].agg(['sum', 'mean', 'count'])
    revenue_by_type = revenue_by_type.sort_values('sum', ascending=True)
    bars = ax.barh(revenue_by_type.index, revenue_by_type['sum'] / 1000, color='steelblue')
    ax.set_xlabel('Total Revenue ($K)')
    ax.set_ylabel('Charter Type')
    ax.set_title('Total Revenue by Charter Type')
    for bar, count in zip(bars, revenue_by_type['count']):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'n={count}', va='center', fontsize=9)
    save_figure(fig, 'b2_revenue_by_charter_type')

    write("### Revenue by Charter Type")
    write()
    write(revenue_by_type.round(2).to_markdown())

    # B3: Revenue by party size (scatter + regression)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(df['# of Pax'], df['Total'], alpha=0.5, edgecolor='none')
    # Add regression line
    mask = ~(df['# of Pax'].isna() | df['Total'].isna())
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df.loc[mask, '# of Pax'], df.loc[mask, 'Total']
    )
    x_line = np.array([df['# of Pax'].min(), df['# of Pax'].max()])
    ax.plot(x_line, intercept + slope * x_line, 'r-',
            label=f'R² = {r_value**2:.3f}')
    ax.set_xlabel('Party Size (# of Pax)')
    ax.set_ylabel('Total Revenue ($)')
    ax.set_title('Revenue vs Party Size')
    ax.legend()
    save_figure(fig, 'b3_revenue_vs_party_size')

    write("**Revenue vs Party Size Regression:**")
    write(f"- Slope: ${slope:.2f} per guest")
    write(f"- R²: {r_value**2:.3f}")
    write()

    # B4: Revenue per guest by charter type
    fig, ax = plt.subplots(figsize=FIGSIZE)
    order = df.groupby('charter_type')['revenue_per_guest'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='charter_type', y='revenue_per_guest', order=order, ax=ax)
    ax.set_xlabel('Charter Type')
    ax.set_ylabel('Revenue per Guest ($)')
    ax.set_title('Revenue per Guest by Charter Type')
    plt.xticks(rotation=45, ha='right')
    save_figure(fig, 'b4_revenue_per_guest_by_type')

    # B5: Revenue by vessel
    fig, ax = plt.subplots(figsize=FIGSIZE)
    revenue_by_vessel = df.groupby('vessel')['Total'].agg(['sum', 'mean', 'count'])
    revenue_by_vessel = revenue_by_vessel.sort_values('sum', ascending=True)
    bars = ax.barh(revenue_by_vessel.index, revenue_by_vessel['sum'] / 1000, color='teal')
    ax.set_xlabel('Total Revenue ($K)')
    ax.set_ylabel('Vessel')
    ax.set_title('Total Revenue by Vessel')
    for bar, count in zip(bars, revenue_by_vessel['count']):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'n={count}', va='center', fontsize=9)
    save_figure(fig, 'b5_revenue_by_vessel')

    write("### Revenue by Vessel")
    write()
    write(revenue_by_vessel.round(2).to_markdown())


# =============================================================================
# Section C: Temporal Patterns
# =============================================================================
def temporal_analysis(df: pd.DataFrame):
    """Generate temporal pattern analysis."""
    section_header("C. Temporal Patterns")

    # C1: Bookings by month (all years combined)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    monthly = df.groupby(['year', 'month']).size().reset_index(name='count')
    for year in sorted(df['year'].unique()):
        year_data = monthly[monthly['year'] == year]
        ax.plot(year_data['month'], year_data['count'], marker='o', label=str(year))
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Bookings')
    ax.set_title('Monthly Booking Volume by Year')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(title='Year')
    save_figure(fig, 'c1_monthly_bookings_by_year')

    # C2: Bookings by day of week
    fig, ax = plt.subplots(figsize=FIGSIZE)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(day_order)
    ax.bar(range(7), day_counts.values, color='steelblue')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Bookings')
    ax.set_title('Bookings by Day of Week')
    save_figure(fig, 'c2_bookings_by_day_of_week')

    write("### Bookings by Day of Week")
    write()
    write(day_counts.to_markdown())

    # C3: Monthly revenue trends
    fig, ax = plt.subplots(figsize=FIGSIZE)
    monthly_rev = df.groupby(['year', 'month'])['Total'].sum().reset_index()
    for year in sorted(df['year'].unique()):
        year_data = monthly_rev[monthly_rev['year'] == year]
        ax.plot(year_data['month'], year_data['Total'] / 1000, marker='o', label=str(year))
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Revenue ($K)')
    ax.set_title('Monthly Revenue by Year')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(title='Year')
    save_figure(fig, 'c3_monthly_revenue_by_year')

    # C4: Seasonal heatmap (month x day of week)
    fig, ax = plt.subplots(figsize=(12, 8))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = df.groupby(['month', 'day_of_week']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(columns=day_order)
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Month')
    ax.set_title('Booking Frequency: Month vs Day of Week')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_yticklabels([month_names[m - 1] for m in heatmap_data.index])
    save_figure(fig, 'c4_seasonal_heatmap')


# =============================================================================
# Section D: Lead Time Analysis
# =============================================================================
def lead_time_analysis(df: pd.DataFrame):
    """Generate lead time analysis."""
    section_header("D. Lead Time Analysis")

    # Filter valid lead times
    df_valid = df[df['lead_time_days'] >= 0].copy()

    write(f"**Lead Time Statistics** (n={len(df_valid)}):")
    write(f"- Mean: {df_valid['lead_time_days'].mean():.1f} days")
    write(f"- Median: {df_valid['lead_time_days'].median():.1f} days")
    write(f"- Std: {df_valid['lead_time_days'].std():.1f} days")
    write(f"- Min: {df_valid['lead_time_days'].min():.0f} days")
    write(f"- Max: {df_valid['lead_time_days'].max():.0f} days")
    write()

    # D1: Lead time distribution
    fig, ax = plt.subplots(figsize=FIGSIZE)
    df_valid['lead_time_days'].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.axvline(df_valid['lead_time_days'].median(), color='red', linestyle='--',
               label=f"Median: {df_valid['lead_time_days'].median():.0f} days")
    ax.set_xlabel('Lead Time (Days)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Booking Lead Time')
    ax.legend()
    save_figure(fig, 'd1_lead_time_distribution')

    # D2: Lead time by charter type
    fig, ax = plt.subplots(figsize=FIGSIZE)
    order = df_valid.groupby('charter_type')['lead_time_days'].median().sort_values(ascending=False).index
    sns.boxplot(data=df_valid, x='charter_type', y='lead_time_days', order=order, ax=ax)
    ax.set_xlabel('Charter Type')
    ax.set_ylabel('Lead Time (Days)')
    ax.set_title('Lead Time by Charter Type')
    plt.xticks(rotation=45, ha='right')
    save_figure(fig, 'd2_lead_time_by_charter_type')

    # D3: Lead time vs revenue
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(df_valid['lead_time_days'], df_valid['Total'], alpha=0.4, edgecolor='none')
    ax.set_xlabel('Lead Time (Days)')
    ax.set_ylabel('Total Revenue ($)')
    ax.set_title('Lead Time vs Revenue')
    # Add regression
    mask = ~(df_valid['lead_time_days'].isna() | df_valid['Total'].isna())
    if mask.sum() > 10:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_valid.loc[mask, 'lead_time_days'], df_valid.loc[mask, 'Total']
        )
        x_line = np.array([0, df_valid['lead_time_days'].max()])
        ax.plot(x_line, intercept + slope * x_line, 'r-',
                label=f'R² = {r_value**2:.3f}')
        ax.legend()
    save_figure(fig, 'd3_lead_time_vs_revenue')


# =============================================================================
# Section E: Booking Channel Analysis
# =============================================================================
def channel_analysis(df: pd.DataFrame):
    """Generate booking channel analysis."""
    section_header("E. Booking Channel Analysis")

    channel_stats = df.groupby('booking_channel').agg({
        'Total': ['sum', 'mean', 'count']
    }).round(2)
    channel_stats.columns = ['Total Revenue', 'Avg Revenue', 'Count']
    channel_stats = channel_stats.sort_values('Total Revenue', ascending=False)

    write("### Booking Channel Statistics")
    write()
    write(channel_stats.to_markdown())

    # E1: Revenue by channel
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Total revenue
    ax = axes[0]
    channel_rev = df.groupby('booking_channel')['Total'].sum().sort_values(ascending=True)
    ax.barh(channel_rev.index, channel_rev.values / 1000, color='steelblue')
    ax.set_xlabel('Total Revenue ($K)')
    ax.set_title('Total Revenue by Booking Channel')

    # Volume
    ax = axes[1]
    channel_count = df['booking_channel'].value_counts().sort_values(ascending=True)
    ax.barh(channel_count.index, channel_count.values, color='teal')
    ax.set_xlabel('Number of Bookings')
    ax.set_title('Booking Volume by Channel')

    plt.tight_layout()
    save_figure(fig, 'e1_channel_analysis')


# =============================================================================
# Section F: Correlation Analysis
# =============================================================================
def correlation_analysis(df: pd.DataFrame):
    """Generate correlation analysis."""
    section_header("F. Correlation Analysis")

    numeric_cols = ['Total', '# of Pax', 'lead_time_days', 'revenue_per_guest',
                    'charter_duration_hours', 'Total Tax', 'Processing Fees']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    corr_matrix = df[numeric_cols].corr()

    write("### Correlation Matrix")
    write()
    write(corr_matrix.round(3).to_markdown())

    # F1: Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title('Correlation Heatmap of Numeric Features')
    save_figure(fig, 'f1_correlation_heatmap')

    # Key correlations
    write()
    write("**Key Correlations:**")
    write(f"- Total vs # of Pax: {corr_matrix.loc['Total', '# of Pax']:.3f}")
    if 'lead_time_days' in corr_matrix.columns:
        write(f"- Total vs Lead Time: {corr_matrix.loc['Total', 'lead_time_days']:.3f}")


# =============================================================================
# Main
# =============================================================================
def main():
    """Run complete EDA pipeline."""
    global _report_file

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Open report file
    with open(REPORT_PATH, 'w') as f:
        _report_file = f

        write("# Casco Bay Custom Charters - EDA Report")
        write()

        # Load data
        print("Loading data...")
        df = load_booking_data()
        write(f"**Loaded {len(df)} valid bookings**")

        # Run all analysis sections
        data_overview(df)
        revenue_analysis(df)
        temporal_analysis(df)
        lead_time_analysis(df)
        channel_analysis(df)
        correlation_analysis(df)

        # Summary
        section_header("Summary")
        write(f"- **Total bookings analyzed:** {len(df)}")
        write(f"- **Total revenue:** ${df['Total'].sum():,.2f}")
        write(f"- **Average booking value:** ${df['Total'].mean():,.2f}")

        _report_file = None

    print(f"\nReport saved to: {REPORT_PATH.absolute()}")
    print(f"Figures saved to: {OUTPUT_DIR.absolute()}")


if __name__ == '__main__':
    main()
