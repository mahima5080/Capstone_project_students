import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load Dataset
print("Loading Students dataset...")
df = pd.read_csv("Students.csv")

# Clean column headers
df.columns = df.columns.str.strip()

# 2. Define Features and Target
categorical_cols = [
    "Stream",
    "AI_Tools_Used",
    "Do_Professors_Allow_Use",
    "Preferred_AI_Tool",
    "Willing_to_Pay_for_Access",
    "Device_Used",
    "Internet_Access",
]

numerical_cols = [
    "Year_of_Study",
    "Daily_Usage_Hours",
    "Trust_in_AI_Tools",
    "Awareness_Level",
]

target_col = "Impact_on_Grades"

X = df[categorical_cols + numerical_cols]
y = df[target_col]

# 3. Data Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False),
            categorical_cols,
        ),
    ]
)

# Transform features to numerical matrices
X_processed = preprocessor.fit_transform(X)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 4. Build TensorFlow Sequential Neural Network
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),  # Linear output for regression
    ]
)

# Compile Model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 5. Train the Model
print("Training TensorFlow Neural Network...")
model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
)

# 6. Generate Predictions across Entire Dataset
print("Generating predictions...")
predictions = model.predict(X_processed).flatten()

# Add outputs back to DataFrame
df["Predicted_Impact_on_Grades"] = predictions
df["Absolute_Error"] = np.abs(df[target_col] - df["Predicted_Impact_on_Grades"])

# 7. Save Processed CSV for Power BI
output_file = "Student_AI_Impact_Predictions_PowerBI.csv"
df.to_csv(output_file, index=False)
print(f"Model training complete! File saved as '{output_file}'")