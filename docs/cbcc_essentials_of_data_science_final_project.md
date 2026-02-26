# Final Project Proposal
## Essentials of Data Science

# Data-Driven Revenue Optimization and Demand Forecasting
### Casco Bay Custom Charters

---

## 1. Project Overview

This project analyzes reservation data from Casco Bay Custom Charters (CBCC), a private charter company operating in Portland, Maine. The objective is to use data science techniques to understand revenue drivers, booking behavior, and seasonal demand patterns in order to generate actionable business insights.

The analysis will combine:

- Exploratory Data Analysis (EDA)
- Statistical modeling
- Time series analysis
- Business interpretation and recommendations

---

## 2. Primary Research Questions

### Q1: What drives charter revenue?
- Which charter types generate the highest revenue?
- How does party size affect total booking value?
- What is revenue per vessel hour?
- How much do add-ons increase total booking revenue?

### Q2: What are the seasonal and demand patterns?
- What are weekly and monthly booking trends?
- When does demand accelerate during the season?
- How do booking curves differ by charter type?

### Q3: How far in advance do guests book?
- Lead time distribution across charter types
- Lead time vs total spend
- Lead time vs cancellation behavior

### Q4: Can we build a statistical model to estimate revenue?
- Use multiple linear regression to model total booking revenue
- Evaluate goodness of fit
- Interpret coefficients to understand revenue drivers

---

## 3. Dataset Description

The reservation dataset may include:

- Booking date
- Departure date
- Charter type
- Vessel
- Party size
- Total revenue
- Add-ons (catering, premium beverages, etc.)
- Booking channel
- Cancellation status

Derived features may include:

- Lead time (days between booking and departure)
- Revenue per guest
- Revenue per vessel hour
- Month and week of season
- Day of week

---

## 4. Methodology

### 4.1 Exploratory Data Analysis (EDA)

- Summary statistics
- Distribution plots
- Correlation matrix
- Revenue breakdown by category
- Seasonality visualization

### 4.2 Feature Engineering

- Calculate lead time
- Create categorical encodings
- Create time-based variables
- Construct revenue efficiency metrics

### 4.3 Revenue Modeling

Model total booking revenue using:

- Multiple Linear Regression

Evaluate:

- R-squared
- Adjusted R-squared
- Residual analysis
- Interpretation of coefficients

### 4.4 Time Series Forecasting

- Weekly booking volume trends
- Moving averages
- Seasonal decomposition
- Forecast next season demand

---

## 5. Business Applications

The results of this project can support:

- Pricing strategy decisions
- Marketing timing and targeting
- Vessel utilization planning
- Add-on revenue optimization
- Staffing and operational forecasting

---

## 6. Expected Outcomes

The project aims to:

- Identify the strongest revenue drivers
- Quantify add-on revenue uplift
- Understand booking lead-time patterns
- Forecast seasonal demand
- Provide data-backed recommendations for revenue optimization

---

## 7. Potential Extensions

- Revenue per available charter hour (RevPACH)
- Channel performance analysis

---

## 8. Conclusion

This project demonstrates the application of core data science techniques to a real-world seasonal tourism business. By combining exploratory analysis, statistical modeling, and business interpretation, the study provides both academic rigor and practical value.

The ultimate goal is to move CBCC toward more data-driven decision-making in pricing, marketing, and operational planning.

