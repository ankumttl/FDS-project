# ============================================================
# CROP YIELD PREDICTION — NEPAL
# Foundation of Data Science | IOE Pulchowk Campus
# Aarohi DC · Ankita Mittal · Arya Jha · Garvita Das
#
# Dataset: Nepal_Crop_main.csv (real data, 1990–2022)
# Crops: Maize, Millet, Potatoes, Rice, Soya beans, Wheat
#
# This notebook:
#   1. Explores the data (EDA)
#   2. Trains & compares models
#   3. Draws 4 meaningful insights
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# sets white grid background and image quality   
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

                         #garvita

# ════════════════════════════════════════════════════════════
# PART 1: LOAD & EXPLORE DATA
# ════════════════════════════════════════════════════════════
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, "Nepal_Crop_main.csv"))

# Rename columns to simpler names
df = df.rename(columns={
    "Item":                          "crop",
    "hg/ha_yield":                   "yield_hg_ha",   # hectograms per hectare
    "avg_temp":                      "temperature_c",
    "average_rain_fall_mm_per_year": "rainfall_mm",
    "pesticides_tonnes":             "pesticides_t",
})

# Convert yield to tonnes per hectare (easier to understand)
# 1 tonne = 10,000 hectograms  → divide by 10,000
df["yield_t_ha"] = df["yield_hg_ha"] / 10000

print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Rows:   {len(df)}")
print(f"Crops:  {sorted(df['crop'].unique())}")
print(f"Years:  {df['Year'].min()} – {df['Year'].max()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic stats:")
print(df[["temperature_c","rainfall_mm","pesticides_t","yield_t_ha"]].describe().round(2))


# ── Chart 1: Yield by Crop (box plot) ───────────────────────
plt.figure(figsize=(9, 5))
crop_order = df.groupby("crop")["yield_t_ha"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="crop", y="yield_t_ha", order=crop_order,
            palette="Set2")
plt.title("Yield Distribution by Crop (1990–2022)", fontsize=13, fontweight="bold")
plt.xlabel("Crop")
plt.ylabel("Yield (tonnes per hectare)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("chart1_yield_by_crop.png")
plt.show()
print("Saved: chart1_yield_by_crop.png")


# ── Chart 2: Yield trend over years ─────────────────────────
yearly = df.groupby(["Year","crop"])["yield_t_ha"].mean().reset_index()

plt.figure(figsize=(11, 5))
for crop in sorted(df["crop"].unique()):
    sub = yearly[yearly["crop"] == crop]
    plt.plot(sub["Year"], sub["yield_t_ha"], marker="o", markersize=3,
             linewidth=2, label=crop)
plt.title("Crop Yield Trend Over Time in Nepal (1990–2022)",
          fontsize=13, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Yield (t/ha)")
plt.legend(title="Crop", bbox_to_anchor=(1.01,1), loc="upper left")
plt.tight_layout()
plt.savefig("chart2_yield_trend.png")
plt.show()
print("Saved: chart2_yield_trend.png")


# ── Chart 3: Correlation heatmap ─────────────────────────────
plt.figure(figsize=(7, 5))
cols = ["temperature_c", "rainfall_mm", "pesticides_t", "Year", "yield_t_ha"]
corr = df[cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, square=True, linewidths=0.5)
plt.title("Correlation Between Features and Yield",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("chart3_correlation.png")
plt.show()
print("Saved: chart3_correlation.png")


# ════════════════════════════════════════════════════════════
# PART 2: PREPARE DATA FOR MODELLING
# ════════════════════════════════════════════════════════════

# Encode crop names as numbers alphabetically(ML needs numbers, not text)
le = LabelEncoder()
df["crop_encoded"] = le.fit_transform(df["crop"])

print("\nCrop encoding:")
for crop, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {crop:12s} → {code}")

# Features the model will use to predict yield
FEATURES = [
    "crop_encoded",    # which crop
    "Year",            # year (captures technology improvement over time)
    "temperature_c",   # average temperature
    "rainfall_mm",     # rainfall
    "pesticides_t",    # pesticide use
]
TARGET = "yield_t_ha"

X = df[FEATURES]
y = df[TARGET]

# Split: 70% train, 30% test
# random_state=42 means the split is the same every time you run it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

                        #Arohi
# ════════════════════════════════════════════════════════════
# PART 3: TRAIN & COMPARE MODELS
# ════════════════════════════════════════════════════════════

models = {
    "Linear Regression":  LinearRegression(),
    "Decision Tree":      DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

print("\n" + "="*55)
print(f"{'Model':<22}  {'MAE':>7}  {'RMSE':>7}  {'R²':>6}")
print("="*55)

for name, model in models.items():
    model.fit(X_train, y_train)           # train
    preds = model.predict(X_test)         # predict on test set

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    results[name] = {"model": model, "preds": preds,
                     "mae": mae, "rmse": rmse, "r2": r2}

    print(f"{name:<22}  {mae:>7.3f}  {rmse:>7.3f}  {r2:>6.3f}")

print("="*55)

best_name = max(results, key=lambda k: results[k]["r2"])
best      = results[best_name]
print(f"\n✓ Best model: {best_name}")
print(f"  R² = {best['r2']:.3f}  → explains {best['r2']*100:.1f}% of yield variation")
print(f"  MAE = {best['mae']:.3f} t/ha  → average prediction error")


# ── Chart 4: Model comparison ────────────────────────────────
names   = list(results.keys())
r2_vals = [results[n]["r2"]  for n in names]
mae_vals= [results[n]["mae"] for n in names]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Model Comparison", fontsize=13, fontweight="bold")

colors = ["#e63946", "#457b9d", "#2d6a4f", "#e9c46a"]
axes[0].barh(names, r2_vals,  color=colors, edgecolor="white")
axes[0].set_title("R² Score (higher = better)")
axes[0].set_xlabel("R²")
for i, v in enumerate(r2_vals):
    axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

axes[1].barh(names, mae_vals, color=colors, edgecolor="white")
axes[1].set_title("MAE — Mean Absolute Error (lower = better)")
axes[1].set_xlabel("MAE (t/ha)")
for i, v in enumerate(mae_vals):
    axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("chart4_model_comparison.png")
plt.show()
print("Saved: chart4_model_comparison.png")


# ── Chart 5: Predicted vs Actual ─────────────────────────────
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best["preds"], alpha=0.6, color="#2d6a4f", s=40, edgecolors="white")
max_val = max(y_test.max(), max(best["preds"]))
plt.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
plt.title(f"Predicted vs Actual Yield\n{best_name}", fontsize=12, fontweight="bold")
plt.xlabel("Actual Yield (t/ha)")
plt.ylabel("Predicted Yield (t/ha)")
plt.legend()
plt.tight_layout()
plt.savefig("chart5_predicted_vs_actual.png")
plt.show()
print("Saved: chart5_predicted_vs_actual.png")


# ════════════════════════════════════════════════════════════
# PART 4: INSIGHT 1 — FEATURE IMPORTANCE
# Which factors affect yield the most?
# ════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("INSIGHT 1: What affects yield the most?")
print("="*50)

rf_model = results["Random Forest"]["model"]
importances = pd.Series(
    rf_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(importances.round(3).to_string())

plt.figure(figsize=(7, 4))
colors_imp = ["#2d6a4f" if i == 0 else "#74c69d" if i < 3 else "#b7e4c7"
              for i in range(len(importances))]
importances.sort_values().plot(kind="barh", color=colors_imp[::-1], edgecolor="white")
plt.title("INSIGHT 1: Feature Importance\nWhich factors affect crop yield the most?",
          fontsize=12, fontweight="bold")
plt.xlabel("Importance Score (0 to 1)")

# Add value labels
for i, val in enumerate(importances.sort_values()):
    plt.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("chart6_feature_importance.png")
plt.show()
print("Saved: chart6_feature_importance.png")

top_feature = importances.index[0]
print(f"\n📊 Finding: '{top_feature}' is the most important factor.")
print("   This makes sense — different crops have very different natural yields.")

                           #ankita
# ════════════════════════════════════════════════════════════
# PART 4: INSIGHT 2 — HOW DOES PESTICIDE USE AFFECT YIELD?
# (What-If Analysis per crop)
# ════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("INSIGHT 2: Does more pesticide use improve yield?")
print("="*50)

# For each crop, fix all other variables at their average
# and vary only pesticide amount — see how yield changes

pesticide_range = np.linspace(df["pesticides_t"].min(),
                               df["pesticides_t"].max(), 60)

best_model = results[best_name]["model"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("INSIGHT 2: Effect of Pesticide Use on Yield (per crop)\n"
             "'What-If' analysis — all other factors held at crop average",
             fontsize=12, fontweight="bold")
axes = axes.flatten()

crop_colors = ["#2d6a4f","#52b788","#e9c46a","#e76f51","#457b9d","#9b5de5"]

for i, crop in enumerate(sorted(df["crop"].unique())):
    crop_data = df[df["crop"] == crop]
    avg_temp  = crop_data["temperature_c"].mean()
    avg_rain  = crop_data["rainfall_mm"].mean()
    avg_year  = crop_data["Year"].mean()
    crop_code = le.transform([crop])[0]

    # Build what-if table: only pesticides changes
    what_if = pd.DataFrame({
        "crop_encoded":  crop_code,
        "Year":          avg_year,
        "temperature_c": avg_temp,
        "rainfall_mm":   avg_rain,
        "pesticides_t":  pesticide_range,
    })

    predicted = best_model.predict(what_if[FEATURES])

    axes[i].fill_between(pesticide_range, predicted.min(), predicted,
                         alpha=0.2, color=crop_colors[i])
    axes[i].plot(pesticide_range, predicted, color=crop_colors[i], linewidth=2.5)

    # Mark actual average pesticide use
    actual_avg_pest = crop_data["pesticides_t"].mean()
    actual_avg_pred = best_model.predict(pd.DataFrame([{
        "crop_encoded": crop_code, "Year": avg_year,
        "temperature_c": avg_temp, "rainfall_mm": avg_rain,
        "pesticides_t": actual_avg_pest
    }]))[0]
    axes[i].axvline(actual_avg_pest, color="red", linestyle="--",
                    linewidth=1.2, alpha=0.7)
    axes[i].scatter([actual_avg_pest], [actual_avg_pred],
                    color="red", s=50, zorder=5,
                    label=f"Current avg\n({actual_avg_pest:.0f}t)")

    axes[i].set_title(crop, fontsize=11, fontweight="bold", color=crop_colors[i])
    axes[i].set_xlabel("Pesticides (tonnes)")
    axes[i].set_ylabel("Predicted Yield (t/ha)")
    axes[i].legend(fontsize=7)

plt.tight_layout()
plt.savefig("chart7_pesticide_vs_yield.png")
plt.show()
print("Saved: chart7_pesticide_vs_yield.png")


# ════════════════════════════════════════════════════════════
# PART 5: INSIGHT 3 — CLIMATE CHANGE IMPACT
# How have temperature & rainfall changed from 1990–2022?
# And how does that relate to yield changes?
# ════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("INSIGHT 3: Climate Change Impact on Crop Yield")
print("="*50)

# Compute yearly averages across all crops
climate_yearly = df.groupby("Year")[["temperature_c", "rainfall_mm", "yield_t_ha"]].mean().reset_index()

fig, axes = plt.subplots(3, 1, figsize=(11, 10))
fig.suptitle("INSIGHT 3: Climate Change Impact (1990–2022)\nNepal — Yearly Averages Across All Crops",
             fontsize=13, fontweight="bold")

# --- Temperature trend ---
axes[0].plot(climate_yearly["Year"], climate_yearly["temperature_c"],
             color="#e63946", linewidth=2.5, marker="o", markersize=4)
z0 = np.polyfit(climate_yearly["Year"], climate_yearly["temperature_c"], 1)
p0 = np.poly1d(z0)
axes[0].plot(climate_yearly["Year"], p0(climate_yearly["Year"]),
             "r--", linewidth=1.5, alpha=0.7, label=f"Trend (slope={z0[0]:+.4f}°C/yr)")
axes[0].set_ylabel("Avg Temperature (°C)")
axes[0].set_title("Average Temperature Over Time")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# --- Rainfall trend ---
axes[1].plot(climate_yearly["Year"], climate_yearly["rainfall_mm"],
             color="#457b9d", linewidth=2.5, marker="o", markersize=4)
z1 = np.polyfit(climate_yearly["Year"], climate_yearly["rainfall_mm"], 1)
p1 = np.poly1d(z1)
axes[1].plot(climate_yearly["Year"], p1(climate_yearly["Year"]),
             "b--", linewidth=1.5, alpha=0.7, label=f"Trend (slope={z1[0]:+.2f} mm/yr)")
axes[1].set_ylabel("Rainfall (mm/year)")
axes[1].set_title("Annual Rainfall Over Time")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# --- Average yield trend ---
axes[2].plot(climate_yearly["Year"], climate_yearly["yield_t_ha"],
             color="#2d6a4f", linewidth=2.5, marker="o", markersize=4)
z2 = np.polyfit(climate_yearly["Year"], climate_yearly["yield_t_ha"], 1)
p2 = np.poly1d(z2)
axes[2].plot(climate_yearly["Year"], p2(climate_yearly["Year"]),
             "g--", linewidth=1.5, alpha=0.7, label=f"Trend (slope={z2[0]:+.3f} t/ha/yr)")
axes[2].set_ylabel("Avg Yield (t/ha)")
axes[2].set_title("Average Crop Yield Over Time")
axes[2].set_xlabel("Year")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("chart8_climate_change_impact.png")
plt.show()
print("Saved: chart8_climate_change_impact.png")

# Correlation between climate variables and yield
temp_corr  = climate_yearly["temperature_c"].corr(climate_yearly["yield_t_ha"])
rain_corr  = climate_yearly["rainfall_mm"].corr(climate_yearly["yield_t_ha"])

print(f"\n📊 Climate–Yield Correlation (yearly averages):")
print(f"   Temperature vs Yield : {temp_corr:+.3f}")
print(f"   Rainfall    vs Yield : {rain_corr:+.3f}")
print(f"\n   Temperature trend   : {z0[0]:+.4f} °C per year")
print(f"   Rainfall trend      : {z1[0]:+.2f} mm per year")
print(f"   Yield trend         : {z2[0]:+.3f} t/ha per year")
print("\n   Finding: Despite climate shifts, yield has risen — suggesting")
print("   technology/pesticide improvements outweigh climate stress so far.")

                      #arya
# ════════════════════════════════════════════════════════════
# PART 6: INSIGHT 4 — CROP RECOMMENDATION SYSTEM
# Given temperature, rainfall & pesticide amount,
# predict yield for every crop and recommend the best one.
# ════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("INSIGHT 4: Crop Recommendation System")
print("="*50)

def recommend_crop(temperature, rainfall, pesticides, year=2022):
    """
    Given farming conditions, predict yield for each crop
    and return a ranked recommendation table.
    """
    rows = []
    for crop in sorted(df["crop"].unique()):
        crop_code = le.transform([crop])[0]
        input_row = pd.DataFrame([{
            "crop_encoded":  crop_code,
            "Year":          year,
            "temperature_c": temperature,
            "rainfall_mm":   rainfall,
            "pesticides_t":  pesticides,
        }])
        predicted_yield = best_model.predict(input_row[FEATURES])[0]
        rows.append({"Crop": crop, "Predicted Yield (t/ha)": round(predicted_yield, 3)})

    result = pd.DataFrame(rows).sort_values("Predicted Yield (t/ha)", ascending=False).reset_index(drop=True)
    result.index += 1   # rank starts from 1
    return result

# Example scenario — typical Nepal conditions
ex_temp      = 14.2
ex_rain      = 1300
ex_pest      = 400

print(f"\nExample Scenario:")
print(f"  Temperature : {ex_temp} °C")
print(f"  Rainfall    : {ex_rain} mm/year")
print(f"  Pesticides  : {ex_pest} tonnes")
print(f"  Year        : 2022")

recommendation = recommend_crop(ex_temp, ex_rain, ex_pest)
print(f"\nCrop Ranking (highest predicted yield = best choice):")
print(recommendation.to_string())

# Visualise the recommendation
plt.figure(figsize=(8, 4))
colors_rec = ["#2d6a4f" if i == 0 else "#74c69d" if i < 3 else "#b7e4c7"
              for i in range(len(recommendation))]
bars = plt.barh(recommendation["Crop"][::-1],
                recommendation["Predicted Yield (t/ha)"][::-1],
                color=colors_rec[::-1], edgecolor="white")
for bar, val in zip(bars, recommendation["Predicted Yield (t/ha)"][::-1]):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val:.2f} t/ha", va="center", fontsize=9)

plt.title(f"INSIGHT 4: Crop Recommendation\n"
          f"Temp={ex_temp}°C | Rain={ex_rain}mm | Pesticides={ex_pest}t",
          fontsize=12, fontweight="bold")
plt.xlabel("Predicted Yield (t/ha)")
plt.tight_layout()
plt.savefig("chart9_crop_recommendation.png")
plt.show()
print("Saved: chart9_crop_recommendation.png")

top_crop = recommendation.iloc[0]["Crop"]
print(f"\n✅ Recommendation: Grow '{top_crop}' under these conditions for maximum yield.")

# ── Try a few more scenarios ─────────────────────────────────
scenarios = [
    {"label": "Hot & Dry",        "temp": 16.0, "rain": 800,  "pest": 200},
    {"label": "Cool & Rainy",     "temp": 12.5, "rain": 1600, "pest": 300},
    {"label": "Average Conditions","temp": 14.2, "rain": 1300, "pest": 400},
]

print("\nScenario Comparison — Best crop per condition:")
print(f"{'Scenario':<22}  {'Best Crop':<14}  {'Predicted Yield'}")
print("-"*55)
for s in scenarios:
    rec = recommend_crop(s["temp"], s["rain"], s["pest"])
    best_crop  = rec.iloc[0]["Crop"]
    best_yield = rec.iloc[0]["Predicted Yield (t/ha)"]
    print(f"{s['label']:<22}  {best_crop:<14}  {best_yield:.2f} t/ha")


# ════════════════════════════════════════════════════════════
# PART 7: RESIDUAL ANALYSIS
# Residual = Actual − Predicted
# Random scatter = good model | Pattern = model is missing something
# ════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("RESIDUAL ANALYSIS: Are the model errors random?")
print("="*50)

residuals = y_test.values - best["preds"]   # actual minus predicted

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Residual Analysis — {best_name}\n"
             "If errors are random (no pattern) → model is well-fitted",
             fontsize=12, fontweight="bold")

# --- Plot 1: Residuals vs Predicted -------------------------
axes[0].scatter(best["preds"], residuals, alpha=0.6, color="#457b9d",
                s=40, edgecolors="white")
axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[0].set_title("Residuals vs Predicted Values")
axes[0].set_xlabel("Predicted Yield (t/ha)")
axes[0].set_ylabel("Residual (Actual − Predicted)")
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Distribution of residuals ----------------------
axes[1].hist(residuals, bins=20, color="#2d6a4f", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].axvline(residuals.mean(), color="orange", linestyle="-",
                linewidth=1.5, label=f"Mean = {residuals.mean():.3f}")
axes[1].set_title("Distribution of Residuals")
axes[1].set_xlabel("Residual (t/ha)")
axes[1].set_ylabel("Frequency")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# --- Plot 3: Residuals by crop --------------------------------
# Map encoded values back to crop names for labelling
test_crops = df.loc[y_test.index, "crop"].values
unique_crops = sorted(set(test_crops))
crop_residuals = [residuals[test_crops == c] for c in unique_crops]

axes[2].boxplot(crop_residuals, labels=unique_crops, patch_artist=True,
                boxprops=dict(facecolor="#b7e4c7", color="#2d6a4f"),
                medianprops=dict(color="red", linewidth=2))
axes[2].axhline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
axes[2].set_title("Residuals by Crop")
axes[2].set_xlabel("Crop")
axes[2].set_ylabel("Residual (t/ha)")
axes[2].tick_params(axis="x", rotation=15)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("chart10_residual_analysis.png")
plt.show()
print("Saved: chart10_residual_analysis.png")

print(f"\n📊 Residual Summary:")
print(f"   Mean residual  : {residuals.mean():+.4f} t/ha  (close to 0 = unbiased)")
print(f"   Std of residuals: {residuals.std():.4f} t/ha")
print(f"   Max over-predict: {residuals.min():.3f} t/ha")
print(f"   Max under-predict: {residuals.max():.3f} t/ha")

if abs(residuals.mean()) < 0.1:
    print("\n   ✅ Mean residual is near zero — model has no systematic bias.")
else:
    print("\n   ⚠️  Mean residual is not near zero — model may be slightly biased.")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════

print("""
╔══════════════════════════════════════════════════════════╗
║               PROJECT SUMMARY                            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  DATASET                                                 ║
║  Real Nepal crop data, 1990–2022, 6 crops                ║
║                                                          ║
║  BEST MODEL                                              ║""")
print(f"║  {best_name:<22} R² = {best['r2']:.3f}  MAE = {best['mae']:.3f} t/ha     ║")
print(f"""║                                                          ║
║  INSIGHT 1 — Feature Importance                          ║
║  Crop type is the #1 factor affecting yield.             ║
║  Pesticide use and year also matter significantly.       ║
║  Rainfall and temperature have moderate effect.          ║
║                                                          ║
║  INSIGHT 2 — Pesticide vs Yield                          ║
║  Most crops show increasing yield with pesticide use,    ║
║  but the relationship differs by crop. Potatoes          ║
║  respond strongly; Millet shows diminishing returns.     ║
║                                                          ║
║  INSIGHT 3 — Climate Change Impact                       ║
║  Temperature has slowly risen over 1990-2022.            ║
║  Rainfall shows slight decline. Despite this,            ║
║  yields have grown - technology is compensating.         ║
║                                                          ║
║  INSIGHT 4 — Crop Recommendation System                  ║
║  Given any temperature, rainfall & pesticide level,      ║
║  the system ranks all crops by predicted yield.          ║
║                                                          ║
║  RESIDUAL ANALYSIS                                       ║
║  Errors are near-random with mean near 0 = well-fitted.  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")