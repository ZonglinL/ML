import pandas as pd
from sklearn.model_selection import train_test_split

# Read CSV
df = pd.read_csv("Reviews.csv")

# Check the label column
print(df.head())

label_col = "Score"

# Find the minimum class count
min_count = df[label_col].value_counts().min()

# Sample min_count items from each class
df_balanced = df.groupby(label_col).sample(n=min_count, random_state=42)

# Shuffle the result if needed
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

df = df_balanced[["Text", "Score"]]

# Stratified train-test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,             # 20% for test
    stratify=df["Score"],      # Preserve label distribution
    random_state=42            # For reproducibility
)

dist = train_df["Score"].value_counts()
percent = train_df["Score"].value_counts(normalize=True)
print(pd.DataFrame({"count": dist, "percent": percent}))



dist = test_df["Score"].value_counts()
percent = test_df["Score"].value_counts(normalize=True)
print(pd.DataFrame({"count": dist, "percent": percent}))


train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)