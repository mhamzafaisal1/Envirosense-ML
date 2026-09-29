import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("updated_with_synthetic_crops.csv", parse_dates=["date"])

# Define healthy ranges for rice
healthy_conditions = {
    "N": (70, 100), "P": (30, 60), "K": (30, 50),
    "temperature": (20, 30), "humidity": (60, 85),
    "ph": (5.5, 7.5), "rainfall": (200, 300),
    "air quality": (30, 40), "light intensity": (400, 600)
}

# Label 'state' based on healthy ranges for classifier training
def determine_state(row, conditions):
    for feature, (low, high) in conditions.items():
        if not (low <= row[feature] <= high):
            return "needs improvement"
    return "healthy"

df['state'] = df.apply(lambda row: determine_state(row, healthy_conditions), axis=1)

# Define all possible values for the categorical columns
all_planting_times = ["Early April", "Early June", "Late April", "Mid October", "Late May"]
all_next_season_crops = ["Corn", "Soybean", "Wheat", "Barley"]

# Separate LabelEncoders for each categorical column
planting_time_encoder = LabelEncoder().fit(all_planting_times)
crop_encoder = LabelEncoder().fit(all_next_season_crops)

# Encode categorical columns in the dataframe
df['ideal_planting_time_encoded'] = planting_time_encoder.transform(df['ideal_planting_time'])
df['next_season_crop_encoded'] = crop_encoder.transform(df['next_season_crop'])

# Drop unnecessary columns after encoding
df = df.drop(columns=['date', 'ideal_planting_time', 'next_season_crop'], errors='ignore')

# Prepare data for Field Condition Classification
X_state = df.drop(columns=['state', 'yield', 'ideal_planting_time_encoded', 'next_season_crop_encoded'])
y_state = df['state']
scaler_state = StandardScaler()
X_state_scaled = scaler_state.fit_transform(X_state)
X_state_train, X_state_test, y_state_train, y_state_test = train_test_split(X_state_scaled, y_state, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_state_train_smote, y_state_train_smote = smote.fit_resample(X_state_train, y_state_train)

# Train classifiers for field condition
state_model = RandomForestClassifier(random_state=42, class_weight="balanced")
state_model.fit(X_state_train_smote, y_state_train_smote)

logistic_model_state = LogisticRegression(random_state=42, class_weight="balanced")
logistic_model_state.fit(X_state_train_smote, y_state_train_smote)

# Prepare data for Yield Regression
X_yield = df.drop(columns=['yield', 'state', 'ideal_planting_time_encoded', 'next_season_crop_encoded'])
y_yield = df['yield']
scaler_yield = StandardScaler()
X_yield_scaled = scaler_yield.fit_transform(X_yield)
X_yield_train, X_yield_test, y_yield_train, y_yield_test = train_test_split(X_yield_scaled, y_yield, test_size=0.2, random_state=42)

# Train regressors for yield
yield_model = RandomForestRegressor(random_state=42)
yield_model.fit(X_yield_train, y_yield_train)

linear_model_yield = LinearRegression()
linear_model_yield.fit(X_yield_train, y_yield_train)

# Prepare data for Ideal Planting Time Classification
X_planting_time = df.drop(columns=['ideal_planting_time_encoded', 'state', 'yield', 'next_season_crop_encoded'])
y_planting_time = df['ideal_planting_time_encoded']
X_planting_time_scaled = scaler_state.fit_transform(X_planting_time)
X_planting_train, X_planting_test, y_planting_train, y_planting_test = train_test_split(X_planting_time_scaled, y_planting_time, test_size=0.2, random_state=42)

planting_time_model = RandomForestClassifier(random_state=42, class_weight="balanced")
planting_time_model.fit(X_planting_train, y_planting_train)

logistic_model_planting_time = LogisticRegression(random_state=42, multi_class="multinomial", solver="lbfgs", class_weight="balanced")
logistic_model_planting_time.fit(X_planting_train, y_planting_train)

# Prepare data for Next Season Crop Classification
X_crop = df.drop(columns=['next_season_crop_encoded', 'state', 'yield', 'ideal_planting_time_encoded'])
y_crop = df['next_season_crop_encoded']
X_crop_scaled = scaler_state.fit_transform(X_crop)
X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(X_crop_scaled, y_crop, test_size=0.2, random_state=42)

crop_model = RandomForestClassifier(random_state=42, class_weight="balanced")
crop_model.fit(X_crop_train, y_crop_train)

logistic_model_crop = LogisticRegression(random_state=42, multi_class="multinomial", solver="lbfgs", class_weight="balanced")
logistic_model_crop.fit(X_crop_train, y_crop_train)

# Save models and scalers
joblib.dump(state_model, 'state_model.pkl')
joblib.dump(logistic_model_state, 'logistic_state_model.pkl')
joblib.dump(yield_model, 'yield_model.pkl')
joblib.dump(linear_model_yield, 'linear_yield_model.pkl')
joblib.dump(planting_time_model, 'planting_time_model.pkl')
joblib.dump(logistic_model_planting_time, 'logistic_planting_model.pkl')
joblib.dump(crop_model, 'crop_model.pkl')
joblib.dump(logistic_model_crop, 'logistic_crop_model.pkl')
joblib.dump(scaler_state, 'scaler_state.pkl')
joblib.dump(scaler_yield, 'scaler_yield.pkl')

print("Models and scalers have been saved as .pkl files!")
