---
author: "Kyle Jones"
date_published: "July 23, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/structural-break-in-u-s-43427d18d9da"
---

# Structural Break in U.S. Housing Market Dynamics After March 2020 using Regression Discontinuity The U.S. housing market shifted in 2020. Listings that once lingered on
the market began to vanish in days. Median days on market dropped...

### Structural Break in U.S. Housing Market Dynamics After March 2020 using Regression Discontinuity in Python
The U.S. housing market shifted in 2020. Listings that once lingered on the market began to vanish in days. Median days on market dropped and stayed low. This is obviously due to COVID, right? Or maybe interest rates? Or something else?

Realtor.com tracks monthly medians of "days on market" (DOM), listing prices, inventory counts, and more for the US. Our goal is to understand whether the market changed suddenly in March 2020 and what might explain that shift.

### Step 1: A Visual Clue
We start by plotting national median DOM over time. Using monthly data aggregated across all U.S. counties:

``` 
plt.plot(monthly_dom["month"], monthly_dom["median_days_on_market"], color="black")
plt.axvline(pd.to_datetime("2020-03-01"), linestyle="--", color="red")
```

The result is striking. DOM declines sharply starting in March 2020.


### Regression Discontinuity in Time (RDiT)
To test whether this drop is statistically meaningful, we use a regression discontinuity approach. We define:

- `t`: months relative to March 2020 (t = 0)
- `post`: 1 if date is March 2020 or later
- `t_post`: interaction to allow slope to change after cutoff

``` 
model = smf.ols("median_days_on_market ~ t + post + t_post", data=monthly_dom).fit()
print(model.summary())
```

#### Result:
- `post = -21.0`, p \< 0.001
- DOM dropped 21 days immediately after March 2020
- Slope did not change significantly


### Was It Just Interest Rates?
I pulled 30-year mortgage rates from FRED using `pandas_datareader` .

``` 
mortgage = web.DataReader("MORTGAGE30US", "fred", start, end)
model = smf.ols("median_days_on_market ~ t + post + t_post + rate", data=df).fit()
```

#### Result:
- `post = -20.3`, still significant
- `rate` is not significant

Interest rates fell in 2020 --- but they do **not** explain the sharp break in DOM.

### Could It Be Inventory?
We test whether changes in housing supply explain the shift.

#### a. Active Listing Count
We add `active_listing_count` (total homes on the market) to the model:

``` 
model = smf.ols("median_days_on_market ~ t + post + t_post + active_listing_count", data=df).fit()
```

**Result:**

- `post = -16.5`, still significant
- `active_listing_count` not significant

Active inventory doesn't explain the shift.

#### b. New Listing Count
Next we test `new_listing_count`, a proxy for seller behavior (willingness to move):

``` 
model = smf.ols("median_days_on_market ~ t + post + t_post + new_listing_count", data=df).fit()
```

**Result:**

- `new_listing_count = -0.0001`, highly significant (p \< 0.001)
- `post = -20.7`, still significant
- R² = 0.71 (much higher than previous models)

Fewer new listings clearly drove part of the DOM drop. Sellers stayed put. The pipeline of new homes dried up. But that alone does not explain the magnitude or persistence of the shift.

### Supply Shock Meets Behavioral Change
The March 2020 break in the housing market reflects both a collapse in new listings and a structural shift in buyer behavior. Homes sold faster not just because of falling rates or low inventory. They sold faster because the market itself changed.

Buyers became more aggressive. Sellers became more cautious. And the entire system changed.

### Files and Reproducibility
All plots, datasets, and models can be reproduced using:

- [[https://www.realtor.com/research/data/](https://www.realtor.com/research/data/) `RDC_Inventory_Core_Metrics_County_History.csv`]
- FRED series `MORTGAGE30US`

This analysis was done entirely in Python using open public data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas_datareader.data as web

CUTOFF = pd.to_datetime("2020-03-01")

def load_monthly_aggregates(filepath, usecols, metrics):
    data = {}
    for chunk in pd.read_csv(filepath, usecols=usecols, chunksize=50000):
        chunk = chunk.dropna(subset=metrics)
        chunk["month"] = pd.to_datetime(chunk["month_date_yyyymm"].astype(str), format="%Y%m")
        grouped = chunk.groupby("month").agg({m: ("median" if m == "median_days_on_market" else "sum") for m in metrics})
        for date, row in grouped.iterrows():
            data.setdefault(date, {m: [] for m in metrics})
            for m in metrics:
                data[date][m].append(row[m])
    return pd.DataFrame({
        "month": list(data.keys()),
        **{m: [pd.Series(v[m]).median() if m == "median_days_on_market" else pd.Series(v[m]).sum()
              for v in data.values()] for m in metrics}
    }).sort_values("month")

def add_rdit_variables(df):
    df["t"] = (df["month"] - CUTOFF).dt.days // 30
    df["post"] = (df["t"] >= 0).astype(int)
    df["t_post"] = df["t"] * df["post"]
    return df

def run_rdit_regression(df, formula):
    model = smf.ols(formula, data=df).fit()
    print(model.summary())
    return model

def plot_time_series(df, x, y, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x], df[y], color="black", linewidth=2)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_rdit_fit(df, x, y, fitted, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x], df[y], color="gray", alpha=0.5, label="Observed")
    plt.plot(df[x], df[fitted], color="black", linewidth=2, label="Fitted")
    plt.axvline(0, color="red", linestyle="--", label="March 2020")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title(title)
    plt.xlabel("Months since March 2020")
    plt.ylabel("Median Days on Market")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# --- Load and plot baseline DOM data ---
baseline = load_monthly_aggregates(
    filepath="RDC_Inventory_Core_Metrics_County_History.csv",
    usecols=["month_date_yyyymm", "median_days_on_market"],
    metrics=["median_days_on_market"]
)

plot_time_series(baseline, "month", "median_days_on_market",
                 "National Median Days on Market Over Time", "national_dom_trend.png")

baseline = add_rdit_variables(baseline)
model = run_rdit_regression(baseline, "median_days_on_market ~ t + post + t_post")

baseline["fitted"] = model.predict(baseline)
plot_rdit_fit(baseline, "t", "median_days_on_market", "fitted",
              "Regression Discontinuity in Time: Days on Market", "regression_discontinuity_dom.png")

# --- Add interest rate as control ---
start, end = baseline["month"].min(), baseline["month"].max()
mortgage = web.DataReader("MORTGAGE30US", "fred", start, end).reset_index()
mortgage.columns = ["month", "rate"]
mortgage["month"] = mortgage["month"].dt.to_period("M").dt.to_timestamp()
mortgage = mortgage.dropna()

df_rate = baseline.merge(mortgage, on="month", how="left")
df_rate = add_rdit_variables(df_rate)
run_rdit_regression(df_rate, "median_days_on_market ~ t + post + t_post + rate")

# --- Add active listings as control ---
df_active = load_monthly_aggregates(
    filepath="RDC_Inventory_Core_Metrics_County_History.csv",
    usecols=["month_date_yyyymm", "median_days_on_market", "active_listing_count"],
    metrics=["median_days_on_market", "active_listing_count"]
)
df_active = add_rdit_variables(df_active)
run_rdit_regression(df_active, "median_days_on_market ~ t + post + t_post + active_listing_count")

# --- Add new listings as control ---
df_new = load_monthly_aggregates(
    filepath="RDC_Inventory_Core_Metrics_County_History.csv",
    usecols=["month_date_yyyymm", "median_days_on_market", "new_listing_count"],
    metrics=["median_days_on_market", "new_listing_count"]
)
df_new = add_rdit_variables(df_new)
run_rdit_regression(df_new, "median_days_on_market ~ t + post + t_post + new_listing_count")
```
