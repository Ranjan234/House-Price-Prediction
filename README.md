# 🏠 House Price Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A supervised machine learning project that predicts residential house prices using regression models trained on a realistic synthetic dataset of 50,000 records and 19 features.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset Description](#-dataset-description)
- [Dataset Columns](#-dataset-columns)
- [Project Workflow](#-project-workflow)
- [Feature Engineering](#-feature-engineering)
- [Models Used](#-models-used)
- [Model Evaluation](#-model-evaluation)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technologies Used](#-technologies-used)
- [Future Improvements](#-future-improvements)

---

## 📖 Overview

This project builds and evaluates multiple regression models to predict residential house prices. The dataset is a realistic synthetic dataset designed to simulate real-world real estate scenarios, covering key residential property attributes such as physical size, location, age, amenities, and socio-economic factors.

The best model — **Random Forest Regressor** — was selected based on R², MAE, and RMSE metrics, validated with 5-fold cross-validation, and saved as a deployable `.pkl` file.

---

## 📂 Dataset Description

| Property        | Detail                                 |
| --------------- | -------------------------------------- |
| Source          | Synthetic (realistic simulation)       |
| Records         | 50,000 rows                            |
| Features        | 19 columns                             |
| Target Variable | `price` (house selling price in ₹)     |
| Categorical     | 2 columns (`location`, `income_level`) |
| Numerical       | 17 columns                             |
| Missing Values  | None                                   |
| Duplicates      | None                                   |

---

## 🗂️ Dataset Columns

Below is a detailed description of each feature in the dataset.

### 🔢 Numerical Features

| Column      | Type  | Description                                                                                                                                                                                       | Example Values        |
| ----------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `price`     | int   | **Target variable.** The selling price of the house. Converted from float to int for modeling. Checked for negatives, zeros, outliers, and skewness.                                              | 1,500,000 – 9,500,000 |
| `area`      | float | Total built-up area of the property in square feet. One of the strongest predictors of price. A larger area generally means a higher price. Box plot analysis confirmed the presence of outliers. | 500 – 6,000 sq ft     |
| `bedrooms`  | int   | Number of bedrooms in the house. Used in feature engineering to compute `total_rooms`.                                                                                                            | 1 – 6                 |
| `bathrooms` | int   | Number of bathrooms in the house. Distribution visualized using a histogram. Also used in `total_rooms` computation.                                                                              | 1 – 4                 |
| `floors`    | int   | Number of floors in the property. Used to engineer the binary feature `is_multifloor`.                                                                                                            | 1 – 3                 |
| `age`       | int   | Age of the property in years. Older properties may have lower prices. Visualized using a box plot and a line plot (Price vs Age).                                                                 | 0 – 50 years          |

### 🔡 Categorical Features

| Column         | Type   | Description                                                                                                                                                                                                               | Unique Values             |
| -------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `location`     | object | The area or locality where the house is situated. Location is a strong driver of price. Encoded using **Label Encoding** (note: OneHotEncoding would be more appropriate as there is no natural order between locations). | premium, medium,low, etc. |
| `income_level` | object | Socio-economic income level of the neighborhood. Encoded using **Ordinal Encoding** since there is a meaningful rank order: Low < Mid < High. Mapped as `{'low': 0, 'mid': 1, 'high': 2}`.                                | `low`, `mid`, `high`      |

### 🛠️ Engineered Features (Created During Feature Engineering)

| Column           | Type | Description                                                                                             | Formula                  |
| ---------------- | ---- | ------------------------------------------------------------------------------------------------------- | ------------------------ |
| `total_rooms`    | int  | Combined count of bedrooms and bathrooms. Represents the overall room capacity of the house.            | `bedrooms + bathrooms`   |
| `area_per_rooms` | int  | Average area available per room. Measures space efficiency. Cast to int after computation.              | `area / total_rooms`     |
| `is_multifloor`  | int  | Binary flag indicating whether the house has more than one floor. Captures vertical space as a feature. | `1 if floors > 1 else 0` |

---

## 🔄 Project Workflow

```
1. Data Ingestion          →   Load CSV (50,000 rows × 19 features)
2. Data Preprocessing      →   Type checks, duplicate/null validation
3. Exploratory Data Analysis →  Histograms, boxplots, heatmap, scatter plots
4. Feature Engineering     →   Create total_rooms, area_per_rooms, is_multifloor
5. Encoding                →   Label Encode (location), Ordinal Encode (income_level)
6. Train-Test Split        →   80% train / 20% test (random_state=42)
7. Model Training          →   Linear, Ridge, Gradient Boosting, Random Forest
8. Model Evaluation        →   R², MAE, MSE, RMSE
9. Model Comparison        →   Horizontal bar chart visualization
10. Cross Validation       →   5-Fold CV on best model (Random Forest)
11. Model Saving           →   joblib → house_price_model.pkl
```

---

## ⚙️ Feature Engineering

Three new features were created from existing columns to give models richer signals:

```python
# 1. Total Rooms — combined capacity
df_encoded['total_rooms'] = df_encoded['bedrooms'] + df_encoded['bathrooms']

# 2. Area per Room — space efficiency metric
df_encoded['area_per_rooms'] = (df_encoded['area'] / df_encoded['total_rooms']).astype(int)

# 3. Is Multifloor — binary flag for multi-storey properties
df_encoded['is_multifloor'] = (df_encoded['floors'] > 1).astype(int)
```

---

## 🤖 Models Used

| Model                 | Type             | Key Parameter      |
| --------------------- | ---------------- | ------------------ |
| Linear Regression     | Linear           | Default            |
| Ridge Regression      | Regularized      | `alpha=1.0`        |
| Gradient Boosting     | Ensemble (Boost) | `n_estimators=100` |
| Random Forest ✅ Best | Ensemble (Bag)   | `n_estimators=100` |

---

## 📊 Model Evaluation Metrics

Each model was evaluated using four standard regression metrics:

| Metric | Full Name                    | Interpretation                                           |
| ------ | ---------------------------- | -------------------------------------------------------- |
| R²     | Coefficient of Determination | Closer to 1.0 = better fit                               |
| MAE    | Mean Absolute Error          | Average absolute difference between actual and predicted |
| MSE    | Mean Squared Error           | Penalizes large errors more than MAE                     |
| RMSE   | Root Mean Squared Error      | Same unit as target (price), easier to interpret         |

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

r2   = r2_score(y_test, pred)
mae  = mean_absolute_error(y_test, pred)
mse  = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
```

---

## 🏆 Results

All four models achieved R² scores between **0.99 – 1.00**, indicating high predictive accuracy on this synthetic dataset.

- **Random Forest** was selected as the best model based on highest R² and lowest RMSE.
- **5-Fold Cross-Validation** confirmed model generalizability.
- The final model was serialized using `joblib` for deployment.

```python
import joblib
joblib.dump(best_model, 'house_price_model.pkl')
```

---

## 📁 Project Structure

```
house-price-prediction/
│
├── House_price_prediction.ipynb   # Main Jupyter Notebook
├── house_price_50k.csv            # Dataset (50,000 rows × 19 features)
├── house_price_model.pkl          # Saved best model (Random Forest)
│
├── images/                        # Saved visualizations
│   ├── price_distribution.png
│   ├── price_vs_area.png
│   └── actual_vs_predicted.png
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🛠️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/Ranjan234/House-Price-Prediction.git
cd House-Price-Prediction
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

**Run the notebook**

```bash
jupyter notebook House_price_prediction.ipynb
```

**Or load the saved model for inference**

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('house_price_model.pkl')

# Sample input (must match training feature columns)
sample = pd.DataFrame([{
    'area': 1800,
    'bedrooms': 3,
    'bathrooms': 2,
    'floors': 2,
    'age': 5,
    'total_rooms': 5,
    'area_per_rooms': 360,
    'is_multifloor': 1,
    'location_enc': 1,
    'income_level_enc': 2
}])

predicted_price = model.predict(sample)
print(f"Predicted House Price: ₹{predicted_price[0]:,.0f}")
```

---

## 🧰 Technologies Used

| Tool / Library   | Purpose                             |
| ---------------- | ----------------------------------- |
| Python 3.10+     | Core programming language           |
| Pandas           | Data loading and manipulation       |
| NumPy            | Numerical computations              |
| Matplotlib       | Plotting and visualization          |
| Seaborn          | Statistical data visualization      |
| Scikit-learn     | ML models, encoding, evaluation, CV |
| Joblib           | Model serialization and saving      |
| Jupyter Notebook | Interactive development environment |

**Install all at once:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

---

## 🔮 Future Improvements

- [ ] Replace `LabelEncoder` with `OneHotEncoder` for the `location` column (nominal variable — no rank order)
- [ ] Apply `log1p` transformation on `price` to reduce skewness before training
- [ ] Add `GridSearchCV` or `RandomizedSearchCV` for hyperparameter tuning
- [ ] Include `Lasso Regression` in the model comparison
- [ ] Build a `Streamlit` or `Flask` web app for live price prediction
- [ ] Add SHAP values for model interpretability and feature importance
- [ ] Test on real-world property datasets (e.g., Kaggle housing datasets)

---

## 🙋 Author

**Your Name**

- GitHub: [Ranjan234](https://github.com/Ranjan234)
- LinkedIn: [Soumya ranjan sahoo](https://www.linkedin.com/in/soumyaranjansahoo0/)

---

> ⭐ If you found this project helpful, please give it a star on GitHub!
