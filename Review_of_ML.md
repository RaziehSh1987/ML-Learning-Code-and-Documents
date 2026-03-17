

# 📊 Machine Learning Basics with Python (Jupyter + Scikit-Learn)

> Based on notes and examples from the following tutorial:
> 👉 

---

# 🚀 Machine Learning Workflow

## 🔁 Standard ML Pipeline

```text
1. Import the Data
2. Clean the Data
3. Split Data into Training/Test Sets
4. Create a Model
5. Train the Model
6. Make Predictions
7. Evaluate and Improve
```

---

# 📦 Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
```

---

# ⚙️ Environment Setup

## Install Anaconda + Jupyter

After installing Anaconda, run:

```bash
jupyter notebook
```

➡️ This will open Jupyter in your browser.

---

# 📂 Importing Dataset

## 🔍 Source for datasets

* [https://www.kaggle.com](https://www.kaggle.com)

---

## 📥 Load CSV File

```python
import pandas as pd

df = pd.read_csv('vgsales.csv')
df.shape  # (rows, columns)
```

---

## 📊 Basic Exploration

```python
df.describe()
df.values
```

---

# ⌨️ Jupyter Shortcuts

| Shortcut    | Description                 |
| ----------- | --------------------------- |
| H           | Show all shortcuts          |
| Shift + Tab | Show function documentation |
| Ctrl + /    | Comment/uncomment           |

---

# 🎯 Example Project: Music Recommendation

## 📌 Goal

Predict music genre based on:

* Age
* Gender

---

# 📥 Load Data

```python
import pandas as pd

music_data = pd.read_csv('music.csv')
music_data.head()
```

---

# 🧹 Data Cleaning

Typical steps:

* Remove duplicates
* Handle missing values
* Fix inconsistencies
* Drop irrelevant data

---

# 🧠 Prepare Data

## Define Features (X) and Target (y)

```python
X = music_data.drop(columns=['genre'])
y = music_data['genre']
```

---

# 🤖 Create & Train Model

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, y)
```

---

# 🔮 Make Predictions

```python
predictions = model.predict([[21, 1], [22, 0]])
print(predictions)
```

Expected Output:

```text
['HipHop', 'Dance']
```

---

# 📏 Model Evaluation

## Split Data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## Measure Accuracy

```python
from sklearn.metrics import accuracy_score

model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
print(score)
```

---

# 💾 Persisting Models (Save & Load)

## Save Model

```python
import joblib

joblib.dump(model, 'music-recommender.joblib')
```

---

## Load Model

```python
model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
```

---

# 🌳 Visualizing Decision Trees

## Export Tree

```python
from sklearn import tree

tree.export_graphviz(
    model,
    out_file='music-recommender.dot',
    feature_names=['age', 'gender'],
    class_names=sorted(y.unique()),
    label='all',
    rounded=True,
    filled=True
)
```

---

## 📌 Visualization Steps

1. Install Graphviz
2. Install VSCode extension: **Dot / Graphviz**
3. Open `.dot` file
4. Click **Preview**

---

# 📊 Final Output

The model creates a **decision tree** showing:

* Splits based on age & gender
* Gini impurity
* Class prediction (HipHop, Jazz, Classical, etc.)

---

# 🧠 Key Concepts Summary

| Concept           | Explanation              |
| ----------------- | ------------------------ |
| Features (X)      | Input variables          |
| Target (y)        | Output variable          |
| Train/Test Split  | Prevent overfitting      |
| Accuracy          | Model performance metric |
| Model Persistence | Save trained models      |
| Decision Tree     | Interpretable ML model   |

---

# 🎯 Key Takeaways

* ML is a **step-by-step pipeline**
* Data preparation is **critical**
* Simple models like **Decision Trees** are powerful
* Always evaluate using **test data**
* Save models to reuse in production

---

# 📎 Resources

* Kaggle datasets: [https://www.kaggle.com](https://www.kaggle.com)
* Tutorial video: [https://youtu.be/7eh4d6sabA0](https://youtu.be/7eh4d6sabA0)

---

# 💡 Pro Tip (for interviews)

When explaining ML projects:

👉 Always structure like this:

```text
Problem → Data → Model → Evaluation → Business Impact
```

---
Here is a **clean, GitHub-ready explanation** you can paste directly under your code:

---

## 🌳 `tree.export_graphviz()` – Explanation of Parameters

This function exports a trained Decision Tree model into a `.dot` file, which can be visualized using Graphviz.

```python
tree.export_graphviz(
    model,
    out_file='music-recommender.dot',
    feature_names=['age', 'gender'],
    class_names=sorted(y.unique()),
    label='all',
    rounded=True,
    filled=True
)
```

---

### 📌 Parameters Breakdown

| Parameter                          | Description                                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `model`                            | The trained Decision Tree model that you want to visualize.                                                        |
| `out_file='music-recommender.dot'` | Output file where the tree structure is saved. This `.dot` file can be rendered using Graphviz.                    |
| `feature_names=['age', 'gender']`  | Names of input features used in the model. These appear in the tree nodes to make splits understandable.           |
| `class_names=sorted(y.unique())`   | Names of target classes (labels). `y.unique()` gets all unique values, and `sorted()` ensures consistent ordering. |
| `label='all'`                      | Displays detailed information in each node (e.g., Gini index, samples, value).                                     |
| `rounded=True`                     | Rounds the corners of the boxes in the tree for better visual appearance.                                          |
| `filled=True`                      | Colors the nodes based on class prediction (helps quickly interpret the model).                                    |

---

### 🎯 What This Produces

* A `.dot` file describing the tree structure
* Each node shows:

  * Split condition (e.g., `age <= 30`)
  * Gini impurity
  * Number of samples
  * Class distribution
  * Predicted class

---

### 📊 Example Node Output

```text
age <= 30.5
gini = 0.5
samples = 10
value = [5, 3, 2]
class = HipHop
```

---

### 🚀 How to Visualize

1. Install Graphviz
2. Open `.dot` file in VSCode
3. Use Graphviz preview extension

---

