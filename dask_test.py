import dask.array as da

# Create a large random array with Dask
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# Compute the mean
result = x.mean().compute()
print(f"Mean of large array: {result}")
