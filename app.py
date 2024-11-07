from flask import Flask, jsonify
import dask.array as da
from sqlalchemy import create_engine, text  # Import text for SQL execution

app = Flask(__name__)

# Set up the connection to Postgres
DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)

# Home route to check if Flask is working
@app.route('/')
def home():
    return "Hello, Flask is working!"

# Route to run a Dask computation and store the result in Postgres
@app.route('/compute')
def compute():
    # Create a large random array with Dask
    x = da.random.random((10000, 10000), chunks=(1000, 1000))

    # Compute the mean and convert to a Python float
    result = float(x.mean().compute())

    # Log the result for debugging
    print(f"Computed result: {result}")

    # Store the result in Postgres using SQLAlchemy's text function
    try:
        with engine.begin() as connection:  # Use begin() for handling commits automatically
            query = text("INSERT INTO computations (result) VALUES (:result)")
            connection.execute(query, {"result": result})
            print("Data inserted successfully into the database!")
    except Exception as e:
        print(f"Error inserting into database: {e}")

    return f"Dask computed the mean: {result}"




# Route to retrieve all stored computations
@app.route('/results')
def get_results():
    with engine.connect() as connection:
        # Use text() for the SELECT query as well
        query = text("SELECT * FROM computations")
        results = connection.execute(query).fetchall()

    return jsonify([{"id": row[0], "result": row[1]} for row in results])

if __name__ == "__main__":
    app.run(debug=True)
