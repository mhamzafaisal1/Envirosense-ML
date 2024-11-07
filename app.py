from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Initialize Flask app
app = Flask(__name__)

# Define healthy conditions for rice
healthy_conditions = {
    "N": (70, 100), "P": (30, 60), "K": (30, 50),
    "temperature": (20, 30), "humidity": (60, 85),
    "ph": (5.5, 7.5), "rainfall": (200, 300),
    "air quality": (30, 40), "light intensity": (400, 600)
}

# Define categorical encoders
all_planting_times = ["Early April", "Early June", "Late April", "Mid October", "Late May"]
all_next_season_crops = ["Corn", "Soybean", "Wheat", "Barley"]
planting_time_encoder = LabelEncoder().fit(all_planting_times)
crop_encoder = LabelEncoder().fit(all_next_season_crops)

# Load or train models and scalers
def load_or_train_models():
    try:
        # Load pre-trained models and scalers if they exist
        state_model = joblib.load('state_model.pkl')
        yield_model = joblib.load('yield_model.pkl')
        planting_time_model = joblib.load('planting_time_model.pkl')
        crop_model = joblib.load('crop_model.pkl')
        scaler_state = joblib.load('scaler_state.pkl')
        scaler_yield = joblib.load('scaler_yield.pkl')
    except:
        # Load dataset and train models if not found
        df = pd.read_csv("updated_with_synthetic_crops.csv", parse_dates=["date"])

        # Define state based on healthy ranges
        def determine_state(row, conditions):
            for feature, (low, high) in conditions.items():
                if not (low <= row[feature] <= high):
                    return "needs improvement"
            return "healthy"

        df['state'] = df.apply(lambda row: determine_state(row, healthy_conditions), axis=1)

        # Encode categorical columns
        df['ideal_planting_time_encoded'] = planting_time_encoder.transform(df['ideal_planting_time'])
        df['next_season_crop_encoded'] = crop_encoder.transform(df['next_season_crop'])
        df = df.drop(columns=['date', 'ideal_planting_time', 'next_season_crop'], errors='ignore')

        # Prepare data for training and create models
        X_state = df.drop(columns=['state', 'yield', 'ideal_planting_time_encoded', 'next_season_crop_encoded'])
        y_state = df['state']
        scaler_state = StandardScaler()
        X_state_scaled = scaler_state.fit_transform(X_state)
        smote = SMOTE(random_state=42)
        X_state_train_smote, y_state_train_smote = smote.fit_resample(X_state_scaled, y_state)
        state_model = RandomForestClassifier(random_state=42, class_weight="balanced")
        state_model.fit(X_state_train_smote, y_state_train_smote)

        X_yield = df.drop(columns=['yield', 'state', 'ideal_planting_time_encoded', 'next_season_crop_encoded'])
        y_yield = df['yield']
        scaler_yield = StandardScaler()
        X_yield_scaled = scaler_yield.fit_transform(X_yield)
        yield_model = RandomForestRegressor(random_state=42)
        yield_model.fit(X_yield_scaled, y_yield)

        X_planting_time = df.drop(columns=['ideal_planting_time_encoded', 'state', 'yield', 'next_season_crop_encoded'])
        y_planting_time = df['ideal_planting_time_encoded']
        planting_time_model = RandomForestClassifier(random_state=42, class_weight="balanced")
        planting_time_model.fit(scaler_state.transform(X_planting_time), y_planting_time)

        X_crop = df.drop(columns=['next_season_crop_encoded', 'state', 'yield', 'ideal_planting_time_encoded'])
        y_crop = df['next_season_crop_encoded']
        crop_model = RandomForestClassifier(random_state=42, class_weight="balanced")
        crop_model.fit(scaler_state.transform(X_crop), y_crop)

        # Save models and scalers
        joblib.dump(state_model, 'state_model.pkl')
        joblib.dump(yield_model, 'yield_model.pkl')
        joblib.dump(planting_time_model, 'planting_time_model.pkl')
        joblib.dump(crop_model, 'crop_model.pkl')
        joblib.dump(scaler_state, 'scaler_state.pkl')
        joblib.dump(scaler_yield, 'scaler_yield.pkl')

    return state_model, yield_model, planting_time_model, crop_model, scaler_state, scaler_yield

# Load or train models
state_model, yield_model, planting_time_model, crop_model, scaler_state, scaler_yield = load_or_train_models()

# Define function to generate recommendations
def generate_recommendation(readings):
    # Processing readings to create a DataFrame
    readings_df = pd.DataFrame([readings])
    
    # Scaling and predicting for each model
    state_features = scaler_state.transform(readings_df)
    predicted_state = state_model.predict(state_features)[0]

    yield_features = scaler_yield.transform(readings_df)
    estimated_yield = yield_model.predict(yield_features)[0]

    planting_time_pred = planting_time_model.predict(state_features)[0]
    predicted_planting_time = planting_time_encoder.inverse_transform([planting_time_pred])[0]

    crop_pred = crop_model.predict(state_features)[0]
    predicted_crop = crop_encoder.inverse_transform([crop_pred])[0]

    # Generate recommendations based on predicted state
    recommendations = {"Estimated Yield": f"{estimated_yield:.2f} quintals/acre"}
    if predicted_state == "needs improvement":
        recommendations["Recommendations"] = [
            f"Adjust {feature} (current: {readings[feature]}, optimal: {low}-{high})"
            for feature, (low, high) in healthy_conditions.items()
            if not (low <= readings[feature] <= high)
        ]
    else:
        recommendations["Recommendations"] = ["All conditions are optimal. No action needed."]

    recommendations["Ideal Planting Time"] = predicted_planting_time
    recommendations["Next Season Crop"] = predicted_crop
    
    return recommendations

# Define Flask route for recommendations
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    recommendations = generate_recommendation(data)
    return jsonify(recommendations)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
