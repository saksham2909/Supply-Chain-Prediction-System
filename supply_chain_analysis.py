
# ============================================================
# 1. IMPORTS & SETUP
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,classification_report, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'

print("=" * 60)
print("  SUPPLY CHAIN DELAY PREDICTION & ANALYSIS SYSTEM")
print("=" * 60)


# ============================================================
# 2. LOAD DATA ()
# ============================================================


df = pd.read_csv('supply_chain_data.csv')
print(f"\n[DATA LOADED] Shape: {df.shape}")
print(df.dtypes)


# ============================================================
# 3. DATA PREPROCESSING (CLEANING & TRANSFORMATION FOR CSV )
# ============================================================
print("\n--- DATA PREPROCESSING ---")

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing values
df['Distance_km'].fillna(df['Distance_km'].median(), inplace=True)
df['Shipping_Mode'].fillna(df['Shipping_Mode'].mode()[0], inplace=True)
df['Order_Priority'].fillna(df['Order_Priority'].mode()[0], inplace=True)

# Remove duplicates

before = len(df)
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {before - len(df)}")

# Convert dates

for col in ['Order_Date', 'Shipping_Date', 'Delivery_Date', 'Expected_Delivery_Date']:
    df[col] = pd.to_datetime(df[col])

# Create useful date features

df['Actual_Delivery_Days'] = (df['Delivery_Date'] - df['Order_Date']).dt.days
df['Expected_Delivery_Days'] = (df['Expected_Delivery_Date'] - df['Order_Date']).dt.days
df['Delay_Days'] = df['Actual_Delivery_Days'] - df['Expected_Delivery_Days']
df['Order_Month'] = df['Order_Date'].dt.month
df['Order_DayOfWeek'] = df['Order_Date'].dt.dayofweek

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print(f"\nFinal dataset shape: {df.shape}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n--- FEATURE ENGINEERING ---")

# Distance categories
df['Distance_Category'] = pd.cut(df['Distance_km'],
    bins=[0, 300, 800, 1500, 3000],
    labels=['Short', 'Medium', 'Long', 'Very Long'])

# Peak season (Oct-Dec = festive season in India)
df['Is_Peak_Season'] = df['Order_Month'].apply(lambda x: 1 if x in [10, 11, 12] else 0)

# Weekend order
df['Is_Weekend'] = df['Order_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

print("New features created:")
print(df[['Distance_Category', 'Is_Peak_Season', 'Is_Weekend']].value_counts().head(8))


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n--- EDA ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Supply Chain Delay Analysis - EDA Dashboard', fontsize=16, fontweight='bold', y=1.01)

# Plot 1: Overall Delay Distribution

delay_counts = df['Delayed'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0, 0].pie(delay_counts, labels=['On Time', 'Delayed'], autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11})
axes[0, 0].set_title('Overall Delay Distribution', fontweight='bold')

# Plot 2: Delay by Shipping Mode

delay_by_mode = df.groupby('Shipping_Mode')['Delayed'].mean().sort_values(ascending=False) * 100
bars = axes[0, 1].bar(delay_by_mode.index, delay_by_mode.values,
                       color=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'])
axes[0, 1].set_title('Delay Rate by Shipping Mode (%)', fontweight='bold')
axes[0, 1].set_ylabel('Delay Rate (%)')
axes[0, 1].set_ylim(0, 70)
for bar, val in zip(bars, delay_by_mode.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,  f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 3: Monthly Delay Trend

monthly = df.groupby('Order_Month')['Delayed'].mean() * 100
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
axes[0, 2].plot(range(1, 13), monthly.values, marker='o', linewidth=2.5,
                color='#e74c3c', markersize=7)
axes[0, 2].set_title('Monthly Delay Trend', fontweight='bold')
axes[0, 2].set_xticks(range(1, 13))
axes[0, 2].set_xticklabels(months, rotation=45, fontsize=8)
axes[0, 2].set_ylabel('Delay Rate (%)')
axes[0, 2].fill_between(range(1, 13), monthly.values, alpha=0.15, color='#e74c3c')

# Plot 4: Distance vs Delay

delay_by_dist = df.groupby('Distance_Category')['Delayed'].mean().sort_values(ascending=False) * 100
axes[1, 0].bar(delay_by_dist.index, delay_by_dist.values,
               color=['#e74c3c', '#e67e22', '#f39c12', '#2ecc71'])
axes[1, 0].set_title('Delay Rate by Distance Category', fontweight='bold')
axes[1, 0].set_ylabel('Delay Rate (%)')
for i, (idx, val) in enumerate(delay_by_dist.items()):
    axes[1, 0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=9)

# Plot 5: Delay by Order Priority

delay_by_priority = df.groupby('Order_Priority')['Delayed'].mean().sort_values(ascending=False) * 100
axes[1, 1].barh(delay_by_priority.index, delay_by_priority.values,
                color=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'])
axes[1, 1].set_title('Delay Rate by Order Priority', fontweight='bold')
axes[1, 1].set_xlabel('Delay Rate (%)')
for i, val in enumerate(delay_by_priority.values):
    axes[1, 1].text(val + 0.2, i, f'{val:.1f}%', va='center', fontsize=9)

# Plot 6: Heatmap - Warehouse vs Shipping Mode Delay Rate

heatmap_data = df.pivot_table(values='Delayed', index='Warehouse_Location',
                               columns='Shipping_Mode', aggfunc='mean') * 100
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[1, 2], linewidths=0.5, cbar_kws={'label': 'Delay Rate (%)'})
axes[1, 2].set_title('Delay Rate: Warehouse vs Shipping Mode', fontweight='bold')
axes[1, 2].set_xlabel('')
axes[1, 2].set_ylabel('')
axes[1, 2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('eda_dashboard.png', bbox_inches='tight', dpi=120)
plt.close()
print("EDA dashboard saved: eda_dashboard.png")


# ============================================================
# 6. MACHINE LEARNING
# ============================================================

print("\n--- MACHINE LEARNING ---")

# Encode categoricals

df_ml = df.copy()
le = LabelEncoder()
for col in ['Shipping_Mode', 'Order_Priority', 'Warehouse_Location',
            'Customer_Location', 'Distance_Category']:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# Drop remaining NaN rows for ML

df_ml.dropna(inplace=True)

features = ['Distance_km', 'Shipping_Mode', 'Order_Priority', 'Warehouse_Location',
            'Customer_Location', 'Order_Quantity', 'Order_Value_INR',
            'Actual_Delivery_Days', 'Expected_Delivery_Days', 'Order_Month',
            'Is_Peak_Season', 'Is_Weekend', 'Distance_Category']

X = df_ml[features]
y = df_ml['Delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'accuracy': acc}
    print(f"\n{name}: Accuracy = {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['On Time', 'Delayed']))

# Best model

best_name = max(results, key=lambda k: results[k]['accuracy'])
print(f"\nBEST MODEL: {best_name} ({results[best_name]['accuracy']:.4f})")

# Visualizations FOR ML evaluation
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Machine Learning Model Evaluation', fontsize=14, fontweight='bold')

# Model accuracy comparison

accs = [results[m]['accuracy'] for m in models]
bars = axes[0].bar(models.keys(), accs, color=['#3498db', '#e67e22', '#2ecc71'])
axes[0].set_ylim(0.5, 1.0)
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
axes[0].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

# Confusion matrix for best model

cm = confusion_matrix(y_test, results[best_name]['preds'])
disp = ConfusionMatrixDisplay(cm, display_labels=['On Time', 'Delayed'])
disp.plot(ax=axes[1], cmap='Blues', colorbar=False)
axes[1].set_title(f'Confusion Matrix\n({best_name})')

# Feature importance (Random Forest)

rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True).tail(10)
importances.plot(kind='barh', ax=axes[2], color='#9b59b6')
axes[2].set_title('Top 10 Feature Importances\n(Random Forest)')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('ml_evaluation.png', bbox_inches='tight', dpi=120)
plt.close()
print("ML evaluation saved: ml_evaluation.png")


# ============================================================
# 7. POWER BI READY EXPORT
# ============================================================

powerbi_df = df[['Order_ID', 'Order_Date', 'Shipping_Date', 'Delivery_Date',
                  'Expected_Delivery_Date', 'Warehouse_Location', 'Customer_Location',
                  'Shipping_Mode', 'Order_Priority', 'Distance_km', 'Distance_Category',
                  'Order_Quantity', 'Order_Value_INR', 'Actual_Delivery_Days',
                  'Expected_Delivery_Days', 'Delay_Days', 'Is_Peak_Season',
                  'Is_Weekend', 'Order_Month', 'Delayed']].copy()

powerbi_df['Delay_Status'] = powerbi_df['Delayed'].map({0: 'On Time', 1: 'Delayed'})
powerbi_df.to_csv('supply_chain_powerbi.csv', index=False)
print(f"\nPower BI dataset saved: supply_chain_powerbi.csv ({len(powerbi_df)} rows)")


# ============================================================
# 8. BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("  BUSINESS INSIGHTS SUMMARY")
print("=" * 60)

total_orders = len(df)
delayed_orders = df['Delayed'].sum()
delay_pct = delayed_orders / total_orders * 100
avg_delivery = df['Actual_Delivery_Days'].mean()
avg_delay_days = df[df['Delayed'] == 1]['Delay_Days'].mean()

print(f"\nKEY METRICS:")
print(f"  Total Orders        : {total_orders:,}")
print(f"  Delayed Orders      : {delayed_orders:,} ({delay_pct:.1f}%)")
print(f"  Avg Delivery Time   : {avg_delivery:.1f} days")
print(f"  Avg Delay (when late): {avg_delay_days:.1f} days")

worst_mode = delay_by_mode.idxmax()
best_mode = delay_by_mode.idxmin()
print(f"\nSHIPPING MODE INSIGHTS:")
print(f"  Highest delay rate  : {worst_mode} ({delay_by_mode[worst_mode]:.1f}%)")
print(f"  Lowest delay rate   : {best_mode} ({delay_by_mode[best_mode]:.1f}%)")

peak_delay = df[df['Is_Peak_Season']==1]['Delayed'].mean() * 100
non_peak_delay = df[df['Is_Peak_Season']==0]['Delayed'].mean() * 100
print(f"\nSEASONALITY:")
print(f"  Peak season delay   : {peak_delay:.1f}%")
print(f"  Off-season delay    : {non_peak_delay:.1f}%")

print(f"\nDISTANCE IMPACT:")
for cat, val in delay_by_dist.items():
    print(f"  {cat:12s}: {val:.1f}%")

print(f"\nBEST ML MODEL: {best_name} (Accuracy: {results[best_name]['accuracy']:.2%})")
print("\n[PROJECT COMPLETE] All outputs saved successfully!")
