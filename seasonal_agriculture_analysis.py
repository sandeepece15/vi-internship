import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

CSV_PATH = "2.csv"

df = pd.read_csv(CSV_PATH)

# Basic data-quality checks
print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Preserve the seasonal order used in the project
season_order = ["Kharif", "Rabi", "Zaid"]
df["Season"] = pd.Categorical(
    df["Season"], categories=season_order, ordered=True
)

# -----------------------------
# Seasonal analysis
# -----------------------------
season = (
    df.groupby("Season", observed=True)
      .agg(
          Farms=("Farm_ID", "count"),
          Avg_Yield=("Yield_Tonnes_Ha", "mean"),
          Avg_Profit=("Profit_INR", "mean"),
          Water_Efficiency=(
              "Water_Efficiency_t_per_1000m3", "mean"
          ),
          Avg_Risk=("Disease_Pest_Risk_pct", "mean"),
          Avg_Rainfall=("Rainfall_mm", "mean"),
      )
      .reset_index()
)

print("\nSeason summary:")
print(season.round(2))

plt.figure(figsize=(8, 5))
plt.bar(season["Season"].astype(str), season["Avg_Yield"])
plt.title("Average Yield by Season")
plt.xlabel("Season")
plt.ylabel("Yield (tonnes/ha)")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(season["Season"].astype(str), season["Avg_Profit"])
plt.axhline(0, linewidth=0.9)
plt.title("Average Profit by Season")
plt.xlabel("Season")
plt.ylabel("Profit (INR per farm)")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# -----------------------------
# Crop analysis
# -----------------------------
crop = (
    df.groupby("Crop")
      .agg(
          Avg_Yield=("Yield_Tonnes_Ha", "mean"),
          Avg_Profit=("Profit_INR", "mean")
      )
      .sort_values("Avg_Profit")
)

print("\nCrop summary:")
print(crop.round(2))

plt.figure(figsize=(8, 5))
plt.barh(crop.index, crop["Avg_Profit"])
plt.axvline(0, linewidth=0.9)
plt.title("Average Profit by Crop")
plt.xlabel("Profit (INR per farm)")
plt.tight_layout()
plt.show()

# -----------------------------
# Irrigation analysis
# -----------------------------
irrigation = (
    df.groupby("Irrigation_Method")
      .agg(
          Avg_Yield=("Yield_Tonnes_Ha", "mean"),
          Avg_Profit=("Profit_INR", "mean")
      )
      .sort_values("Avg_Yield", ascending=False)
)

print("\nIrrigation summary:")
print(irrigation.round(2))

plt.figure(figsize=(8, 5))
plt.bar(irrigation.index, irrigation["Avg_Yield"])
plt.title("Average Yield by Irrigation Method")
plt.xlabel("Irrigation Method")
plt.ylabel("Yield (tonnes/ha)")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# -----------------------------
# Relationship analysis
# -----------------------------
corr_cols = [
    "Rainfall_mm", "Avg_Temperature_C", "Humidity_pct",
    "Soil_Moisture_pct", "Fertilizer_kg_ha",
    "Seed_Quality_Score", "Yield_Tonnes_Ha",
    "Profit_INR", "Disease_Pest_Risk_pct"
]
corr = df[corr_cols].corr()
print("\nCorrelations with yield:")
print(corr["Yield_Tonnes_Ha"].sort_values(ascending=False).round(3))

plt.figure(figsize=(9, 6))
plt.imshow(corr.values, aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=60, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Heatmap of Key Variables")
plt.colorbar(label="Correlation")
plt.tight_layout()
plt.show()

# -----------------------------
# Statistical comparison
# -----------------------------
yield_groups = [
    g["Yield_Tonnes_Ha"].dropna().values
    for _, g in df.groupby("Season", observed=True)
]
profit_groups = [
    g["Profit_INR"].dropna().values
    for _, g in df.groupby("Season", observed=True)
]

yield_test = f_oneway(*yield_groups)
profit_test = f_oneway(*profit_groups)

print("\nANOVA - Yield by season:", yield_test)
print("ANOVA - Profit by season:", profit_test)

# Note: missing values are excluded metric-by-metric via dropna().
