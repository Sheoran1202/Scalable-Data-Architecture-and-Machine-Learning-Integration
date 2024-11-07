from sqlalchemy import create_engine, text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle

# Set up the connection to Postgres
DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)

# Function to load data from the database
def load_data_from_db():
    with engine.connect() as connection:
        query = text("SELECT * FROM housing_data")
        results = connection.execute(query)
        df = pd.DataFrame(results.fetchall(), columns=results.keys())
    return df

# Function to retrain the model using RandomForestRegressor
def retrain_model():
    print("Loading data from database...")
    df = load_data_from_db()
    
    # Data preprocessing
    X = df.drop("PRICE", axis=1)
    y = df["PRICE"]
    
    # Apply feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Using RandomForestRegressor instead of LinearRegression
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Retraining completed. Mean Squared Error (MSE): {mse}")
    
    # Save the retrained model
    with open("trained_housing_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Retrained model saved as trained_housing_model.pkl")

# Main function to run the retraining
if __name__ == "__main__":
    retrain_model()
