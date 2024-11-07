import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from datetime import datetime

# Step 1: Data collection (from a CSV for this example)
def collect_data():
    data_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(data_url)
    print(f"Data collected at {datetime.now()}:")
    print(df.head())
    return df

# Step 2: Preprocessing data (this is a basic example, feel free to add more processing steps)
def preprocess_data(df):
    # Let's assume we only care about these columns for now
    df = df[['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'median_house_value']]
    df.columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'PRICE']  # Rename columns for simplicity
    df.dropna(inplace=True)  # Drop missing values
    print("Data preprocessing completed:")
    print(df.head())
    return df

# Step 3: Save the processed data to PostgreSQL
def store_data(df):
    DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
    engine = create_engine(DATABASE_URL)
    df.to_sql('housing_data', engine, if_exists='append', index=False)
    print(f"Data successfully inserted into the database at {datetime.now()}")

if __name__ == "__main__":
    # Run the pipeline steps
    data = collect_data()         # Step 1: Collect
    processed_data = preprocess_data(data)  # Step 2: Preprocess
    store_data(processed_data)    # Step 3: Store in DB
