
# Feature Engineering & Feature Selection

## 📌 Overview
In this project, I applied both **feature engineering** and **feature selection** to improve model performance, reduce noise, and improve generalization on unseen data.

---

# 🧠 Feature Engineering

Feature engineering transforms raw data into more useful inputs for the model.

## 1. Data Cleaning

I cleaned the dataset before training by:
- Handling missing values
- Fixing data types
- Removing duplicates

### Example
```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, 30, None, 40],
    "salary": ["50000", "60000", "70000", "80000"]
})

# Fill missing values
df["age"] = df["age"].fillna(df["age"].median())

# Convert data type
df["salary"] = df["salary"].astype(int)

# Remove duplicates
df = df.drop_duplicates()

print(df)
````

---

## 2. Feature Transformation

I transformed numerical features to make them more suitable for modeling.

### a) Scaling / Normalization

Scaling helps features stay on a similar range.

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.DataFrame({
    "age": [25, 30, 35, 40],
    "income": [30000, 50000, 70000, 90000]
})

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

print(scaled_features)
```

### b) Log Transformation

Log transformation helps reduce skewness in features with very large values.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "sales": [10, 20, 30, 1000]
})

df["log_sales"] = np.log1p(df["sales"])  # log(1 + x)

print(df)
```

> **Skewness** means the data is not balanced around the center. Log transformation can make the distribution more normal-like.

---

## 3. Encoding Categorical Variables

Machine learning models usually need numeric input, so categorical values must be encoded.

### a) One-Hot Encoding

```python
import pandas as pd

df = pd.DataFrame({
    "color": ["red", "blue", "green", "red"]
})

encoded_df = pd.get_dummies(df, columns=["color"])
print(encoded_df)
```

### b) Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.DataFrame({
    "size": ["small", "medium", "large", "small"]
})

encoder = LabelEncoder()
df["size_encoded"] = encoder.fit_transform(df["size"])

print(df)
```

---

## 4. Feature Creation

I created new features from existing columns to help the model capture better patterns.

### a) Time-Based Features

```python
import pandas as pd

df = pd.DataFrame({
    "date": pd.to_datetime(["2024-01-10", "2024-02-15", "2024-03-20"])
})

df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

print(df)
```

### b) Interaction Features

```python
import pandas as pd

df = pd.DataFrame({
    "price": [10, 20, 30],
    "quantity": [2, 3, 4]
})

df["price_x_quantity"] = df["price"] * df["quantity"]

print(df)
```

### c) Domain-Specific Features

Example: rolling average in sales forecasting.

```python
import pandas as pd

df = pd.DataFrame({
    "sales": [100, 120, 130, 150, 170]
})

df["rolling_avg_3"] = df["sales"].rolling(window=3).mean()

print(df)
```

---

# 🎯 Feature Selection

Feature selection keeps only useful features and removes irrelevant or redundant ones.

## 1. Correlation Analysis

I checked correlation between features and removed highly correlated ones.

```python
import pandas as pd

df = pd.DataFrame({
    "height_cm": [160, 170, 180, 190],
    "height_m": [1.60, 1.70, 1.80, 1.90],
    "weight": [55, 70, 80, 90]
})

corr_matrix = df.corr()
print(corr_matrix)
```

If two features are highly correlated, one of them can be removed because they carry similar information.

Example:

* `height_cm`
* `height_m`

These are basically the same feature in different units.

---

## 2. Removing Low-Variance Features

If a feature does not change much, it usually does not help the model.

```python
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

df = pd.DataFrame({
    "feature_1": [1, 1, 1, 1],
    "feature_2": [1, 2, 3, 4]
})

selector = VarianceThreshold(threshold=0.0)
selected_features = selector.fit_transform(df)

print(selected_features)
```

In this example, `feature_1` may be removed because it has no variance.

---

## 3. Model-Based Feature Importance

I used a tree-based model such as Random Forest to measure which features contributed most to prediction.

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

X = pd.DataFrame({
    "age": [25, 30, 35, 40],
    "income": [30000, 50000, 70000, 90000],
    "score": [1, 0, 1, 0]
})

y = [0, 1, 1, 0]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

print(importance_df.sort_values(by="importance", ascending=False))
```

Features with very low importance can be removed to simplify the model and reduce noise.

---

# 🚀 Why This Matters

Using feature engineering and feature selection helped me:

* Improve data quality
* Reduce noise
* Remove redundant information
* Improve model generalization
* Reduce overfitting
* Increase training efficiency

---

# 💡 Key Insight

* **Feature Engineering** = creating and transforming useful inputs
* **Feature Selection** = removing weak, redundant, or noisy features

Both are essential for building a robust machine learning pipeline.

