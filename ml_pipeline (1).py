
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("tips (1).csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# Separate features and target
X = df.drop("tip", axis=1)
y = df["tip"]

numerical_features = ["total_bill", "size"]
categorical_features = ["sex", "smoker", "day", "time"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Create ML Pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# Train Model
model.fit(X_train, y_train)
print("Model Training Completed")

# Evaluation
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Save Model
joblib.dump(model, "tips_model.pkl")
print("Model Saved Successfully")

# Prediction
sample = pd.DataFrame({
    "total_bill": [25.50],
    "sex": ["Male"],
    "smoker": ["No"],
    "day": ["Sun"],
    "time": ["Dinner"],
    "size": [3]
})

prediction = model.predict(sample)
print("Predicted Tip Amount:", round(prediction[0], 2))
