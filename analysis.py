import pandas as pd
from pandas import read_excel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


maize_dt = pd.read_excel("C:/Users/Administrator/OneDrive/Desktop/kenya-maize-flour-price-prediction/Maize prices.xlsx")

#print(maize_dt.head(10))

maize_dt["Retail"] = maize_dt["Retail"].str.replace("/Kg", "", regex=False)
maize_dt["Retail"] = pd.to_numeric(maize_dt["Retail"])

#Dropping columns that are not needed for analysis
maize_dt_2 = maize_dt.drop(columns=["Supply Volume", "Grade", "Sex", "Wholesale"])
print(maize_dt_2.head(10))

#Checks duplicate rows in the dataset
duplicate_rows = maize_dt_2.duplicated()
print(f"Number of duplicate rows: {duplicate_rows.sum()}")
print(f"Duplicate rows:\n{maize_dt_2[duplicate_rows].head(50)}")
remove_duplicates = maize_dt_2.drop_duplicates()
print(f'No of duplicate rows after removing duplicates: {remove_duplicates.duplicated().sum()}')
maize_dt_2.drop_duplicates(inplace=True)
print(f"Checking for duplicates: {maize_dt_2.duplicated().sum()}")

print(f"Missing values in each column:\n{maize_dt_2.isnull().sum()}")

print("Short info about the dataset\n")

print(maize_dt_2.describe())
print(maize_dt_2.info())

maize_dt_2["Date"] = pd.to_datetime(maize_dt_2["Date"], format="%Y-%m-%d")
print(f"\nData type of 'Date' column: {maize_dt_2['Date'].dtypes}\n")
maize_dt_2["Year"] = maize_dt_2["Date"].dt.year
print("First 10 rows of the 'Year' column:\n")
print(maize_dt_2["Year"].head(10))



# Visualizing the retail prices of maize flour over time
sns.lineplot(data=maize_dt_2, x="Date", y="Retail")
plt.title("Retail Prices of Maize Flour Over Time")
plt.xlabel("Date")
plt.ylabel("Retail Price (KES/Kg)")
#plt.show()

print("INFORMATION ABOUT THE DATE AND CLASSIFICATION COLUMNS")
print(maize_dt_2["Date"].min())
print(maize_dt_2["Date"].max())
print(maize_dt_2["Classification"].value_counts())
print(maize_dt_2["Date"].value_counts().sort_index().head(20))
print(maize_dt_2.groupby("Classification")["Date"].nunique().sort_values(ascending=False))
print(maize_dt_2.columns)



# Sort data by date
maize_dt_2 = maize_dt_2.sort_values("Date")

# Convert Date into number of days since the first date
maize_dt_2["Days"] = (
    maize_dt_2["Date"] - maize_dt_2["Date"].min()
).dt.days

print(maize_dt_2[["Date", "Days"]].head(10))

# Select our predictors
X = maize_dt_2[["Classification", "Days"]]

# Select our target
y = maize_dt_2["Retail"]

# Define the preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("classification", OneHotEncoder(handle_unknown="ignore"), ["Classification"])
    ],
    remainder="passthrough"
)

# Transform the predictors
X_encoded = preprocessor.fit_transform(X)

print(X_encoded.shape)

split_index = int(X_encoded.shape[0] * 0.8)

X_train = X_encoded[:split_index]
X_test = X_encoded[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred[:10])
print(y_test[:10].values)
