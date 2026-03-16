# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Exploratory Data Analysis
# **XAI Credit Lens: Default of Credit Card Clients**
#
# This notebook explores the UCI Default of Credit Card Clients dataset,
# examining distributions, correlations, and patterns relevant to credit
# default prediction and fairness analysis.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

PROJECT_ROOT = Path("..").resolve()

# %% [markdown]
# ## 1. Load Data

# %%
raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "credit_default.csv")
processed = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "credit_default_processed.csv")

print(f"Raw dataset: {raw.shape[0]} rows, {raw.shape[1]} columns")
print(f"Processed dataset: {processed.shape[0]} rows, {processed.shape[1]} columns")
print(f"\nDefault rate: {raw['DEFAULT'].mean():.2%}")
raw.head()

# %% [markdown]
# ## 2. Target Variable Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
raw["DEFAULT"].value_counts().plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"])
axes[0].set_title("Default Distribution")
axes[0].set_xticklabels(["No Default (0)", "Default (1)"], rotation=0)
axes[0].set_ylabel("Count")

# Percentage
sizes = raw["DEFAULT"].value_counts()
axes[1].pie(sizes, labels=["No Default", "Default"], autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"], startangle=90)
axes[1].set_title("Default Rate")

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Class imbalance ratio: {sizes[0]/sizes[1]:.1f}:1")

# %% [markdown]
# ## 3. Demographic Analysis

# %%
EDUCATION_MAP = {1: "Grad School", 2: "University", 3: "High School", 4: "Other"}
MARRIAGE_MAP = {1: "Married", 2: "Single", 3: "Other"}
SEX_MAP = {1: "Male", 2: "Female"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Sex
sex_default = raw.groupby("SEX")["DEFAULT"].mean()
sex_default.index = sex_default.index.map(SEX_MAP)
sex_default.plot(kind="bar", ax=axes[0], color=["#3498db", "#e74c3c"])
axes[0].set_title("Default Rate by Sex")
axes[0].set_ylabel("Default Rate")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
for i, v in enumerate(sex_default):
    axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center", fontweight="bold")

# Education
edu_default = raw.groupby("EDUCATION")["DEFAULT"].mean()
edu_default.index = edu_default.index.map(EDUCATION_MAP)
edu_default.plot(kind="bar", ax=axes[1], color=sns.color_palette("husl", 4))
axes[1].set_title("Default Rate by Education")
axes[1].set_ylabel("Default Rate")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15)
for i, v in enumerate(edu_default):
    axes[1].text(i, v + 0.005, f"{v:.1%}", ha="center", fontweight="bold")

# Marriage
mar_default = raw.groupby("MARRIAGE")["DEFAULT"].mean()
mar_default.index = mar_default.index.map(MARRIAGE_MAP)
mar_default.plot(kind="bar", ax=axes[2], color=sns.color_palette("Set2", 3))
axes[2].set_title("Default Rate by Marital Status")
axes[2].set_ylabel("Default Rate")
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)
for i, v in enumerate(mar_default):
    axes[2].text(i, v + 0.005, f"{v:.1%}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "demographic_default_rates.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Age Distribution and Default Risk

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age distribution by default status
for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
    subset = raw[raw["DEFAULT"] == label]
    axes[0].hist(subset["AGE"], bins=30, alpha=0.6, color=color,
                 label=f"{'Default' if label else 'No Default'}", density=True)
axes[0].set_title("Age Distribution by Default Status")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Density")
axes[0].legend()

# Default rate by age group
raw["AGE_BIN"] = pd.cut(raw["AGE"], bins=[20, 25, 30, 35, 40, 50, 60, 80])
age_default = raw.groupby("AGE_BIN", observed=True)["DEFAULT"].mean()
age_default.plot(kind="bar", ax=axes[1], color="#3498db")
axes[1].set_title("Default Rate by Age Group")
axes[1].set_xlabel("Age Group")
axes[1].set_ylabel("Default Rate")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "age_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

raw.drop(columns=["AGE_BIN"], inplace=True)

# %% [markdown]
# ## 5. Payment History Patterns

# %%
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(pay_cols):
    month_names = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
    ct = pd.crosstab(raw[col], raw["DEFAULT"], normalize="index")
    ct.plot(kind="bar", stacked=True, ax=axes[i], color=["#2ecc71", "#e74c3c"], legend=False)
    axes[i].set_title(f"Default Rate by {month_names[i]} Payment Status")
    axes[i].set_xlabel("Payment Status")
    axes[i].set_ylabel("Proportion")

axes[0].legend(["No Default", "Default"], loc="upper left")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "payment_history_patterns.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key Insight:** Higher PAY_X values (longer payment delays) strongly correlate with default.
# Status -1 (paid in full) shows the lowest default rate across all months.

# %% [markdown]
# ## 6. Credit Limit and Bill Amount Analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Credit limit by default
for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
    subset = raw[raw["DEFAULT"] == label]
    axes[0].hist(np.log1p(subset["LIMIT_BAL"]), bins=40, alpha=0.6, color=color,
                 label=f"{'Default' if label else 'No Default'}", density=True)
axes[0].set_title("Credit Limit Distribution (log scale)")
axes[0].set_xlabel("log(Credit Limit)")
axes[0].legend()

# Bill amount trend
bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
months = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
for label, color, name in [(0, "#2ecc71", "No Default"), (1, "#e74c3c", "Default")]:
    subset = raw[raw["DEFAULT"] == label]
    means = [subset[c].mean() for c in bill_cols]
    axes[1].plot(months, means, marker="o", color=color, label=name, linewidth=2)
axes[1].set_title("Average Bill Amount Over 6 Months")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Average Bill Amount (NT$)")
axes[1].legend()

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "credit_bill_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Engineered Features Analysis

# %%
eng_features = [
    "DEBT_TO_INCOME_PROXY", "PAYMENT_RATIO", "UTILIZATION_RATE",
    "MONTHS_DELINQUENT", "AVG_PAYMENT_DELAY", "PAYMENT_TREND",
    "MAX_CONSEC_DELINQUENT", "BALANCE_VOLATILITY",
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, feat in enumerate(eng_features):
    for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
        subset = processed[processed["DEFAULT"] == label]
        data = subset[feat].clip(lower=subset[feat].quantile(0.01),
                                  upper=subset[feat].quantile(0.99))
        axes[i].hist(data, bins=30, alpha=0.5, color=color, density=True,
                     label=f"{'Default' if label else 'No Default'}")
    axes[i].set_title(feat, fontsize=10)
    axes[i].legend(fontsize=8)

plt.suptitle("Engineered Feature Distributions by Default Status", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "engineered_features.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key Insights from Engineered Features:**
# - `MONTHS_DELINQUENT` and `MAX_CONSEC_DELINQUENT` show clear separation between default/no-default
# - `AVG_PAYMENT_DELAY` is strongly right-skewed for defaulters
# - `PAYMENT_RATIO` is lower for defaulters (paying less of their bills)
# - These features should be strong predictors in the model

# %% [markdown]
# ## 8. Correlation Analysis

# %%
# Correlation with target
target_corr = processed.drop(columns=["DEFAULT"]).corrwith(processed["DEFAULT"]).sort_values()

fig, ax = plt.subplots(figsize=(10, 10))
colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in target_corr]
target_corr.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Correlation with Default")
ax.set_xlabel("Pearson Correlation")
ax.axvline(x=0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "target_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Summary Statistics

# %%
summary = processed.groupby("DEFAULT").describe().T
print("Dataset Summary by Default Status:")
print(f"\nTotal samples: {len(processed)}")
print(f"Features: {len(processed.columns) - 1}")
print(f"Default rate: {processed['DEFAULT'].mean():.2%}")
print(f"\nTop correlated features with default:")
top_corr = processed.drop(columns=["DEFAULT"]).corrwith(processed["DEFAULT"]).abs().sort_values(ascending=False)
for feat, corr in top_corr.head(10).items():
    print(f"  {feat}: {corr:.3f}")
