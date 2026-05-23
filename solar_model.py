import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

gen = pd.read_csv("Plant_1_Generation_Data.csv")
weather = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")

gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'], dayfirst=True)
weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], dayfirst=True)

df = pd.merge(gen, weather, on='DATE_TIME')

print("Merged data shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

df['HOUR'] = df['DATE_TIME'].dt.hour
df['DAY'] = df['DATE_TIME'].dt.day
df['MONTH'] = df['DATE_TIME'].dt.month

features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'HOUR', 'DAY', 'MONTH']
target = 'AC_POWER'

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n--- Model Results ---")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print("(R² of 1.0 = perfect, 0.8+ is very good)")

plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Actual', color='blue')
plt.plot(y_pred[:100], label='Predicted', color='orange')
plt.title('Solar Energy Output: Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('AC Power (kW)')
plt.legend()
plt.tight_layout()
plt.savefig('prediction_chart.png')
plt.show()
print("\nChart saved as prediction_chart.png")