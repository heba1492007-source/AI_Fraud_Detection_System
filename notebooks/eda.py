import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("data/creditcard.csv")


# Display the first 5 rows
print("First 5 rows:")
print(df.head())


# Display dataset information
print("\nDataset information:")
df.info()


# Analyze target distribution
print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass distribution (%):")
print(df["Class"].value_counts(normalize=True) * 100)


# Check missing values
print("\nMissing values:")
print(df.isnull().sum())


# Check duplicated rows
print("\nDuplicated rows:")
print(df.duplicated().sum())


# Remove duplicated rows
df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)

print("\nDuplicated rows after removal:")
print(df.duplicated().sum())


# Visualize class distribution
class_counts = df["Class"].value_counts()

plt.figure(figsize=(6, 4))
class_counts.plot(kind="bar")

plt.title("Transaction Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()