# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore the Midwest Survey dataset
#
# In this notebook, we will explore the **Midwest Survey** dataset from
# [skrub](https://skrub-data.org/).
#
# This dataset contains survey responses from people across the United States,
# asking them about their perception of the Midwest region.
#
# The goal is to predict the **Census Region** where a respondent lives,
# based on their survey answers.

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey
from skrub import TableReport
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

dataset = fetch_midwest_survey()

# X contains the features (the survey answers)
X = dataset.X
# y contains the target (the Census Region)
y = dataset.y

# %% [markdown]
# ## Question 1: How many examples are there in the dataset?
#
# Use the `.shape` attribute to find out the number of rows and columns.

# %%
# Display the number of rows and columns
n_rows, n_cols = X.shape
print("Number of examples (rows):", n_rows)
print("Number of features (columns):", n_cols)

# You can also look at the first few rows of the dataset
X.head()

# %% [markdown]
# ## Question 2: What is the distribution of the target?
#
# The target variable `y` tells us the Census Region of each respondent.
# Let's see how many respondents belong to each region.

# %%
# Count how many respondents belong to each region
y_counts = y.value_counts()
print(y_counts)

# %%
# Visualize the target distribution with a bar plot (horizontal)
ax = y_counts.sort_values().plot(
    kind="barh", figsize=(8, 4)
)
ax.set_xlabel("Number of respondents")
ax.set_ylabel("Census Region")
ax.set_title("Distribution of the target (Census Region)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Is the target balanced (roughly the same number of examples per class)
# or imbalanced?

# %% [markdown]
# ## Question 3: What are the features that can be used to predict the target?
#
# Let's look at the column names and their data types.

# %%
# List all column names
print("Feature columns:")
print(X.columns.tolist())

# %%
# Show data types for each column
print("\nData types:")
print(X.dtypes)

# %%
# How many features are numerical? How many are categorical (text)?
n_numeric = (X.dtypes != "object").sum()
n_categ = (X.dtypes == "object").sum()
print(f"\nNumber of numerical features: {n_numeric}")
print(f"Number of categorical (text) features: {n_categ}")

# %%
# Quick automatic report
TableReport(X)

# %% [markdown]
# ## Question 4: Are there any missing values in the dataset?
#
# Missing values can cause problems for machine learning models.
# Let's check if there are any.

# %%
# Check for NaN missing values
na_counts = X.isna().sum()
print("Number of NaN per column:")
print(na_counts)

print("\nTotal number of NaN:", int(na_counts.sum()))

# %% [markdown]
# Missing values can sometimes be encoded differently. Let's look at some
# columns more closely.

# %%
# Look at unique values for the Household_Income column
print("Unique values in Household_Income:")
print(X["Household_Income"].unique())

# %%
# Look at unique values for the Education column
print("\nUnique values in Education:")
print(X["Education"].unique())

# %% [markdown]
# Do you see a special value that could represent missing data?

# %% [markdown]
# ## Question 5: What is the most common answer to
# "How much do you personally identify as a Midwesterner"?
#
# Let's explore this important feature.

# %%
# display the value counts for the column
col_midwest = "How_much_do_you_personally_identify_as_a_Midwesterner"
midwest_counts = X[col_midwest].value_counts()
print(midwest_counts)

# Most common answer
most_common_midwest = midwest_counts.idxmax()
print(
    '\nMost common answer to '
    '"How much do you personally identify as a Midwesterner":',
    most_common_midwest,
)

# %%
# make a bar plot of the results
ax = midwest_counts.sort_values().plot(
    kind="barh", figsize=(8, 4)
)
ax.set_xlabel("Number of respondents")
ax.set_ylabel(col_midwest)
ax.set_title("Distribution of identification as a Midwesterner")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bonus: Explore another feature
#
# Pick another column and explore its distribution.
# For example: `Gender`, `Age`, or one of the
# "Do you consider X state as part of the Midwest" columns.

# %%
# Example: explore the Gender column
col = "Gender"
print(f"Distribution for {col}:")
gender_counts = X[col].value_counts()
print(gender_counts)

ax = gender_counts.sort_values().plot(
    kind="barh", figsize=(6, 4)
)
ax.set_xlabel("Number of respondents")
ax.set_ylabel(col)
ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()
print(y.value_counts())
print(n_numeric, n_categ)
print("Total NaN:", int(X.isna().sum().sum()))
X[col].value_counts()
print(X["How_much_do_you_personally_identify_as_a_Midwesterner"].value_counts())
print(n_numeric, n_categ)
print("Total NaN:", int(X.isna().sum().sum()))
