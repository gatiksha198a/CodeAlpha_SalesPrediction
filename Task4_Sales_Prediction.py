# ============================================================
# TASK 4: Sales Prediction using Python
# CodeAlpha Data Science Internship
# Student: Gatiksha | ID: CA/DF1/101914
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   SALES PREDICTION — CodeAlpha Internship")
print("=" * 55)

# ── 1. LOAD / CREATE DATASET ────────────────────────────────
# If you have the dataset from CodeAlpha, replace with:
# df = pd.read_csv('sales_data.csv')
np.random.seed(42)
n = 200

TV_budget       = np.random.uniform(0.7, 296.4, n)
Radio_budget    = np.random.uniform(0.0, 49.6,  n)
Newspaper_budget= np.random.uniform(0.3, 114.0, n)
platforms       = np.random.choice(['Online', 'Offline', 'Both'], n)
segments        = np.random.choice(['Youth', 'Adults', 'Seniors'], n)

# Sales formula with noise (realistic)
sales = (
    0.045 * TV_budget +
    0.19  * Radio_budget +
    0.002 * Newspaper_budget +
    np.where(platforms == 'Both',   3.0,
    np.where(platforms == 'Online', 1.5, 0.5)) +
    np.where(segments  == 'Adults', 2.0,
    np.where(segments  == 'Youth',  1.0, 0.5)) +
    np.random.normal(0, 1.2, n) +
    2.5
)
sales = np.clip(sales, 1.0, 30.0)

df = pd.DataFrame({
    'TV':               TV_budget,
    'Radio':            Radio_budget,
    'Newspaper':        Newspaper_budget,
    'Platform':         platforms,
    'Target_Segment':   segments,
    'Sales':            np.round(sales, 2)
})

print(f"\n📊 Dataset Shape: {df.shape}")
print(df.head(10))
print("\nBasic Statistics:")
print(df.describe())

# ── 2. EDA ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Advertising Budget vs Sales', fontsize=15, fontweight='bold')
channels = ['TV', 'Radio', 'Newspaper']
colors   = ['#3498DB', '#E74C3C', '#2ECC71']

for i, (ch, col) in enumerate(zip(channels, colors)):
    axes[i].scatter(df[ch], df['Sales'], alpha=0.6, color=col, edgecolors='white', s=50)
    z = np.polyfit(df[ch], df['Sales'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[ch].min(), df[ch].max(), 100)
    axes[i].plot(x_line, p(x_line), 'k--', linewidth=2)
    r2 = np.corrcoef(df[ch], df['Sales'])[0, 1] ** 2
    axes[i].set_title(f'{ch} Budget (R²={r2:.3f})', fontweight='bold')
    axes[i].set_xlabel(f'{ch} Budget ($K)')
    axes[i].set_ylabel('Sales ($K)')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sales_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ EDA saved as 'sales_eda.png'")

# Platform & Segment Impact
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

platform_avg = df.groupby('Platform')['Sales'].mean().sort_values(ascending=False)
axes[0].bar(platform_avg.index, platform_avg.values,
            color=['#3498DB', '#E74C3C', '#2ECC71'], edgecolor='white')
axes[0].set_title('Average Sales by Platform', fontweight='bold')
axes[0].set_ylabel('Average Sales ($K)')
for i, v in enumerate(platform_avg.values):
    axes[0].text(i, v + 0.1, f'${v:.2f}K', ha='center', fontweight='bold')

segment_avg = df.groupby('Target_Segment')['Sales'].mean().sort_values(ascending=False)
axes[1].bar(segment_avg.index, segment_avg.values,
            color=['#9B59B6', '#F39C12', '#1ABC9C'], edgecolor='white')
axes[1].set_title('Average Sales by Target Segment', fontweight='bold')
axes[1].set_ylabel('Average Sales ($K)')
for i, v in enumerate(segment_avg.values):
    axes[1].text(i, v + 0.1, f'${v:.2f}K', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('sales_categorical.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Categorical analysis saved as 'sales_categorical.png'")

# Correlation Heatmap
plt.figure(figsize=(7, 5))
corr = df[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues',
            square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('sales_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Correlation heatmap saved as 'sales_correlation.png'")

# ── 3. PREPROCESSING ────────────────────────────────────────
le_platform = LabelEncoder()
le_segment  = LabelEncoder()
df['Platform_enc']       = le_platform.fit_transform(df['Platform'])
df['Target_Segment_enc'] = le_segment.fit_transform(df['Target_Segment'])

features = ['TV', 'Radio', 'Newspaper', 'Platform_enc', 'Target_Segment_enc']
X = df[features]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. MODEL TRAINING & COMPARISON ──────────────────────────
models = {
    'Linear Regression':     LinearRegression(),
    'Random Forest':         RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':     GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print("\n📊 Model Comparison:")
print("-" * 50)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse    = mean_squared_error(y_test, y_pred)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    results[name] = {'model': model, 'predictions': y_pred,
                     'MSE': mse, 'MAE': mae, 'R2': r2}
    print(f"  {name}:")
    print(f"    R² Score: {r2:.4f} | MAE: {mae:.4f} | RMSE: {np.sqrt(mse):.4f}")
print("-" * 50)

# ── 5. BEST MODEL ───────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]['R2'])
best       = results[best_name]
print(f"\n🏆 Best Model: {best_name} (R²={best['R2']:.4f})")

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best['predictions'], alpha=0.7,
            color='#3498DB', edgecolors='white', s=60, label='Predictions')
min_val = min(y_test.min(), best['predictions'].min())
max_val = max(y_test.max(), best['predictions'].max())
plt.plot([min_val, max_val], [min_val, max_val],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Sales ($K)', fontsize=12)
plt.ylabel('Predicted Sales ($K)', fontsize=12)
plt.title(f'Actual vs Predicted — {best_name}\nR²={best["R2"]:.4f}',
          fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sales_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Actual vs Predicted saved as 'sales_actual_vs_predicted.png'")

# Feature Importance (Gradient Boosting)
gb_model = results['Gradient Boosting']['model']
feat_imp = pd.DataFrame({'Feature': features,
                         'Importance': gb_model.feature_importances_}
                       ).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
bars = plt.bar(feat_imp['Feature'], feat_imp['Importance'],
               color=['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6'],
               edgecolor='white')
plt.title('Feature Importance — Gradient Boosting', fontsize=13, fontweight='bold')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
for bar, imp in zip(bars, feat_imp['Importance']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{imp:.3f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('sales_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature importance saved as 'sales_feature_importance.png'")

# ── 6. BUSINESS INSIGHTS ────────────────────────────────────
print("\n" + "=" * 55)
print("📈 BUSINESS INSIGHTS")
print("=" * 55)
print(f"  1. TV has the {'highest' if feat_imp.iloc[0]['Feature']=='TV' else 'notable'} impact on sales")
print(f"  2. Radio advertising shows strong correlation with sales")
print(f"  3. Best Platform for sales: {platform_avg.idxmax()} (avg ${platform_avg.max():.2f}K)")
print(f"  4. Best Target Segment:     {segment_avg.idxmax()} (avg ${segment_avg.max():.2f}K)")

# Sample prediction
sample_new = pd.DataFrame({'TV': [230.1], 'Radio': [37.8],
                           'Newspaper': [69.2],
                           'Platform_enc': [1], 'Target_Segment_enc': [0]})
sample_scaled = scaler.transform(sample_new)
pred_sales = gb_model.predict(sample_scaled)
print(f"\n  📦 Sample Prediction:")
print(f"     TV=$230K | Radio=$37.8K | Newspaper=$69.2K")
print(f"     Predicted Sales: ${pred_sales[0]:.2f}K")

print("\n" + "=" * 55)
print("   Task 4 Complete! ✅")
print("   Outputs: sales_eda.png, sales_categorical.png,")
print("            sales_correlation.png,")
print("            sales_actual_vs_predicted.png,")
print("            sales_feature_importance.png")
print("=" * 55)
