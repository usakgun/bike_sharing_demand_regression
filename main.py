import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'hour.csv')
df = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
sns.boxplot(x='hr', y='cnt', data=df)
plt.title('Hourly Bike Rentals')
plt.savefig(os.path.join(current_dir, 'hourly_rentals.png'))
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='cnt', data=df)
plt.title('Temperature vs. Rentals')
plt.savefig(os.path.join(current_dir, 'temp_vs_rentals.png'))
plt.show()

categorical_cols = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop(['instant', 'dteday', 'casual', 'registered', 'cnt'], axis=1)
y = df['cnt']

models = {
    'k-NN': Pipeline([
        ('scaler', StandardScaler()), 
        ('knn', KNeighborsRegressor(n_neighbors=5))
    ]),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

scoring = {
    'MAE': 'neg_mean_absolute_error',
    'RMSE': 'neg_root_mean_squared_error',
    'MAPE': 'neg_mean_absolute_percentage_error'
}

cv = KFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    mae = -results['test_MAE'].mean()
    rmse = -results['test_RMSE'].mean()
    mape = -results['test_MAPE'].mean()
    
    print(f"--- {name} ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}\n")