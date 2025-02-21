df = pd.read_csv("/content/used_cars.csv")
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="cividis")
plt.title("Heatmap of Missing Values")
plt.show()
print(df.isnull().sum())
df["milage"] = df["milage"].str.replace("mi.", "", regex=False).str.replace(",", "")
df["milage"] = pd.to_numeric(df["milage"])
df["price"] = df["price"].str.replace("$", "", regex=False).str.replace(",", "")
df["price"] = pd.to_numeric(df["price"])
for col in df:
    if df[col].dtype == "O":
        print(f"Unique value in '{col}':")
        print(pd.unique(df[col]))
        print(f"Count of unique values in '{col}': {len(pd.unique(df[col]))}")
        print()
        print()
unique_values = df["engine"].unique()
unique_values[:50]
df["clean_title"].fillna("No", inplace=True)
df["accident"].fillna("None reported", inplace=True)
df["fuel_type"].replace(["–", "not supported"], np.nan, inplace=True)
df["fuel_type"] = df["fuel_type"].replace(
    {
        "E85 Flex Fuel": 1,
        "Gasoline": 2,
        "Hybrid": 3,
        "Diesel": 4,
        "Plug-In Hybrid": 5,
        np.nan: np.nan,
    }
)

# Apply KNNImputer
imputer = KNNImputer(n_neighbors=3)
df[["fuel_type"]] = imputer.fit_transform(df[["fuel_type"]])

# Convert back to categorical after imputation
df["fuel_type"] = (
    df["fuel_type"]
    .round()
    .replace(
        {
            1: "E85 Flex Fuel",
            2: "Gasoline",
            3: "Hybrid",
            4: "Diesel",
            5: "Plug-In Hybrid",
        }
    )
)
for col in ["fuel_type", "accident", "clean_title"]:
    plt.figure(figsize=(12, 4))  # Set consistent size
    df[col] = df[col].astype("category")  # Convert to categorical type

    # Create the countplot with custom color palette
    ax = sns.countplot(
        y=col, data=df, palette="coolwarm", order=df[col].value_counts().index
    )

    # Add annotations to each bar
    total = len(df[col])
    for p in ax.patches:
        percentage = f"{100 * p.get_width() / total:.1f}%"  # Calculate percentage
        ax.annotate(
            f"{int(p.get_width())} ({percentage})",
            (p.get_width(), p.get_y() + 0.5),
            ha="left",
            va="center",
            fontsize=10,
            color="black",
        )

    # Rotate Y labels if needed and add title
    plt.yticks(rotation=0, fontsize=12)
    plt.xticks(fontsize=12)
    plt.title(f"Count Plot for {col.capitalize()} with Percentages", fontsize=15)

    plt.tight_layout()
    plt.show()


def extract_fuel_type(engine_info):
    if pd.isna(engine_info):
        return np.nan
    if "Gasoline" in engine_info:
        return "Gasoline"
    elif "Hybrid" in engine_info:
        return "Hybrid"
    elif "Flex Fuel" in engine_info or "E85" in engine_info:
        return "Flex Fuel"
    elif "Diesel" in engine_info:
        return "Diesel"
    elif "Electric" in engine_info:
        return "Electric"
    else:
        return "None"


def extract_transmission_type(transmission):
    if "Automatic" in transmission:
        return "Automatic"
    elif "Manual" in transmission:
        return "Manual"
    elif "CVT" in transmission:
        return "CVT"
    elif "DCT" in transmission:
        return "DCT"
    elif "Fixed Gear" in transmission:
        return "Fixed Gear"
    elif "Variable" in transmission:
        return "Variable"
    elif "Single-Speed" in transmission:
        return "Single-Speed"
    else:
        return "None"


def categorize_color(color):
    color = color.lower()
    if any(x in color for x in ["black", "obsidian", "raven", "onyx"]):
        return "Black"
    elif any(x in color for x in ["white", "pearl", "ivory", "frost"]):
        return "White"
    elif any(x in color for x in ["blue", "navy", "aqua", "teal"]):
        return "Blue"
    elif any(x in color for x in ["red", "ruby", "garnet"]):
        return "Red"
    elif any(x in color for x in ["silver", "gray", "grey", "steel"]):
        return "Silver_Gray"
    elif any(x in color for x in ["green"]):
        return "Green"
    elif any(x in color for x in ["yellow", "gold", "orange"]):
        return "Yellow_Orange"
    else:
        return "Other"


import re


def extract_hp(engine_string):
    match = re.search(r"(\d+\.?\d*)HP", engine_string)
    if match:
        return float(match.group(1))
    return None


df["horsepower"] = df["engine"].apply(extract_hp)


def extract_cylinders(engine_string):
    match = re.search(r"(\d+)\s*Cylinder", engine_string)
    if match:
        return int(match.group(1))
    return None


df["Cylinders"] = df["engine"].apply(extract_cylinders)
df["engine"] = df["engine"].apply(extract_fuel_type)
df["transmission_type"] = df["transmission"].apply(extract_transmission_type)
df["categorized_col_ext"] = df["ext_col"].apply(categorize_color)
df["categorized_col_int"] = df["int_col"].apply(categorize_color)
max_year = df["model_year"].max()
df["age"] = max_year - df["model_year"]
df["age"]
df = df.drop("transmission", axis=1)
df = df.drop("model", axis=1)
df = df.drop("ext_col", axis=1)
df = df.drop("int_col", axis=1)
df = df.drop("model_year", axis=1)
for col in df:
    if df[col].dtype == "O":
        print(f"Unique value in '{col}':")
        print(pd.unique(df[col]))
        print(f"Count of unique values in '{col}': {len(pd.unique(df[col]))}")
        print()
        print()
categories = {
    "fuel_type": ["E85 Flex Fuel", "Gasoline", "Hybrid", "Diesel", "Plug-In Hybrid"],
    "accident": ["None reported", "At least 1 accident or damage reported"],
    "clean_title": ["No", "Yes"],
}
for column, cat_list in categories.items():
    encoder = OrdinalEncoder(categories=[cat_list])
    df[column] = encoder.fit_transform(df[[column]]).astype(int)
columns_to_encode = [
    "brand",
    "engine",
    "transmission_type",
    "categorized_col_ext",
    "categorized_col_int",
]
ohe = OneHotEncoder(drop="first", sparse_output=False)

encoded_columns = ohe.fit_transform(df[columns_to_encode])

encoded_df = pd.DataFrame(
    encoded_columns, columns=ohe.get_feature_names_out(columns_to_encode)
)

df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1).drop(
    columns=columns_to_encode
)
df["price"].fillna(df["price"].mean(), inplace=True)
df["Cylinders"].fillna(df["Cylinders"].mean(), inplace=True)
df["horsepower"].fillna(df["horsepower"].mean(), inplace=True)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[["milage", "price", "horsepower"]])
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[["milage", "price"]])
df["price"] = np.log1p(df["price"])
plt.figure(figsize=(10, 6))
plt.hexbin(
    df["age"],
    df["milage"],
    gridsize=40,
    cmap="inferno",
    mincnt=1,
    edgecolors="grey",
    linewidths=0.5,
)
plt.xlabel("Car's Age", fontsize=14)
plt.ylabel("Milage", fontsize=14)
plt.title("Hexbin Plot of Model Year vs Milage", fontsize=16)
colorbar = plt.colorbar(label="Counts", extend="max")
colorbar.set_ticks([0, 10, 20, 50, 100])
colorbar.ax.tick_params(labelsize=10)
plt.show()
sns.pairplot(df[["age", "milage", "price"]], diag_kind="kde", palette="coolwarm")
plt.show()
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap for Numerical Features")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols = ["milage", "age", "price", "horsepower"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

rf_model = RandomForestRegressor(n_estimators=100, random_state=4)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest RMSE: {rmse_rf}")
print(f"Random Forest R² Score: {r2_rf}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (Random Forest)")
plt.plot(
    [min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--"
)  # Diagonal line
plt.show()
import pickle

with open("rf_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)
