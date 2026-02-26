# cbcc-analysis

Exploratory data analysis of the CBCC dataset.

## Setup

```bash
conda env create -f environment.yml
conda activate cbcc-analysis
echo $CONDA_DEFAULT_ENV  # should print: cbcc-analysis
```

## Updating Dependencies

```bash
conda env export --from-history > environment.yml
```

## Usage

```bash
make eda
```

### EDA Output

The script generates summary statistics and visualizations:

- **Data Overview**: Dataset shape, date range, missing values, numeric summaries
- **Revenue Analysis**: Distribution, by charter type, by vessel, revenue per guest (non-cancelled only)
- **Temporal Patterns**: Monthly trends, day-of-week patterns, seasonal heatmap
- **Lead Time Analysis**: Distribution and relationship to revenue
- **Booking Channel Analysis**: Revenue and volume by channel
- **Correlation Analysis**: Heatmap of numeric features
- **Cancellation Analysis**: Cancellation rates, patterns by month/type/channel, lead time comparison

Output:
- Figures: `output/figures/`
- Report: `output/eda_report.md`
