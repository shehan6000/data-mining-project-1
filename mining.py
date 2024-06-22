import pandas as pd
from faker import Faker
import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Faker generator
fake = Faker()

# Function to generate synthetic data
def generate_synthetic_data(num_records):
    data = []
    for _ in range(num_records):
        name = fake.name()
        age = random.randint(18, 80)
        income = round(random.uniform(20000, 150000), 2)
        country = fake.country()
        data.append((name, age, income, country))
    return data

# Generate synthetic data
num_records = 1000
synthetic_data = generate_synthetic_data(num_records)

# Convert to DataFrame
df = pd.DataFrame(synthetic_data, columns=['Name', 'Age', 'Income', 'Country'])

# Data preprocessing and EDA

# Summary statistics
print("Summary statistics:")
print(df.describe())

# Correlation matrix
corr_matrix = df[['Age', 'Income']].corr()
print("\nCorrelation matrix:")
print(corr_matrix)

# EDA: Distribution plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')

plt.subplot(1, 2, 2)
sns.histplot(df['Income'], bins=20, kde=True)
plt.title('Distribution of Income')

plt.tight_layout()
plt.show()

# Clustering analysis

# Clustering using KMeans on age and income
X = df[['Age', 'Income']]

kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Evaluate clustering using silhouette score
silhouette_avg = silhouette_score(X, df['Cluster'])
print(f"\nSilhouette Score for Clustering: {silhouette_avg}")

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['Income'], c=df['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Clustering of Age vs Income')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# Predictive modeling

# Feature engineering (if applicable)
# No explicit feature engineering in this synthetic example

# Predictive modeling: Predict income based on age using Random Forest

X = df[['Age']]
y = df['Income']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=0)
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [None, 5, 10, 15]}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("\nBest Parameters:", best_params)

# Evaluate model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# Visualize predicted vs actual income
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.5)
plt.scatter(X_test, y_pred, color='red', label='Predicted', alpha=0.5)
plt.title('Actual vs Predicted Income')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()
