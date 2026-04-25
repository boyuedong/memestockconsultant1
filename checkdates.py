import pandas as pd

print("Loading only DATE column...")
df = pd.read_csv(
    "/Users/boyuedong/Desktop/new3:11/full_dataset_cleaned.csv",
    usecols=["DATE"]
)
print("Loaded DATE column.")

print("Parsing dates...")
dates = pd.to_datetime(df["DATE"], format="%d/%m/%Y", errors="coerce").dropna()
print("Parsed dates.")

# Keep only 2017 dates
dates_2017 = dates[dates.dt.year == 2017]

actual = pd.DatetimeIndex(dates_2017.dt.normalize().unique()).sort_values()
expected = pd.date_range("2017-01-01", "2017-12-31", freq="D")
missing = expected.difference(actual)

print("Min 2017 date found:", actual.min() if len(actual) else "None")
print("Max 2017 date found:", actual.max() if len(actual) else "None")
print("Unique 2017 dates found:", len(actual))
print("Expected 2017 dates:", len(expected))
print("Missing 2017 dates count:", len(missing))

if len(missing) == 0:
    print("All dates from 2017-01-01 to 2017-12-31 are present.")
else:
    print("Missing dates in 2017:")
    for d in missing:
        print(d.strftime("%Y-%m-%d"))