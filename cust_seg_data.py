import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Setting a seed for reproducibility
np.random.seed(42)

# Define the number of customers
num_customers = 1000

# Helper functions for generating logical data
def generate_age(segment):
    if segment == "Young Adults":
        return np.random.randint(18, 26)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(26, 46)
    else:  # Retired Seniors
        return np.random.randint(60, 81)

def generate_income(segment):
    if segment == "Young Adults":
        return np.random.randint(200000, 600000)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(600000, 2500000)
    else:  # Retired Seniors
        return np.random.randint(300000, 1000000)

def generate_customer_profitability(segment):
    if segment == "Young Adults":
        return np.random.randint(5000, 20000)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(20000, 100000)
    else:  # Retired Seniors
        return np.random.randint(10000, 50000)

def generate_customer_lifetime_value(segment):
    if segment == "Young Adults":
        return np.random.randint(15000, 50000)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(50000, 300000)
    else:  # Retired Seniors
        return np.random.randint(50000, 400000)

def generate_customer_value_till_date(lifetime_value):
    return lifetime_value * np.random.uniform(0.5, 1)

def generate_revenue_last_year(segment):
    if segment == "Young Adults":
        return np.random.randint(5000, 25000)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(15000, 100000)
    else:  # Retired Seniors
        return np.random.randint(10000, 50000)

def generate_net_interest_income_last_year(segment):
    if segment == "Young Adults":
        return np.random.randint(0, 5000)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(10000, 50000)
    else:  # Retired Seniors
        return np.random.randint(0, 10000)

def generate_total_cost(segment):
    if segment == "Young Adults":
        return np.random.randint(2000, 10000)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(5000, 25000)
    else:  # Retired Seniors
        return np.random.randint(3000, 15000)

def generate_total_non_interest_income(segment):
    if segment == "Young Adults":
        return np.random.randint(2000, 10000)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(5000, 30000)
    else:  # Retired Seniors
        return np.random.randint(2000, 20000)

def generate_customer_segment():
    return np.random.choice(["Young Adults", "Mid-Career Professionals", "Retired Seniors"], p=[0.3, 0.5, 0.2])

def generate_customer_region():
    return np.random.choice(["Urban", "Suburban", "Rural"], p=[0.5, 0.3, 0.2])

def generate_account_type(segment):
    if segment == "Young Adults":
        return np.random.choice(["Savings", "Current"])
    elif segment == "Mid-Career Professionals":
        return np.random.choice(["Savings", "Credit Card", "Mortgage", "Loan"])
    else:  # Retired Seniors
        return np.random.choice(["Savings", "Investment"])

def generate_recency_days(segment):
    if segment == "Young Adults":
        return np.random.randint(0, 30)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(0, 60)
    else:  # Retired Seniors
        return np.random.randint(30, 180)

def generate_frequency_last_year(segment):
    if segment == "Young Adults":
        return np.random.randint(20, 100)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(10, 50)
    else:  # Retired Seniors
        return np.random.randint(5, 20)

def generate_monetary_value(segment):
    if segment == "Young Adults":
        return np.random.randint(5000, 100000)
    elif segment == "Mid-Career Professionals":
        return np.random.randint(100000, 5000000)
    else:  # Retired Seniors
        return np.random.randint(500000, 10000000)

# Dataframe creation
data = {
    "CUST_ID": np.arange(1, num_customers + 1),
}

# Generating customer segment
data["CUSTOMER_SEGMENT"] = [generate_customer_segment() for _ in range(num_customers)]

# Populate other columns based on segment
data["AGE"] = [generate_age(segment) for segment in data["CUSTOMER_SEGMENT"]]
data["CUSTOMER_PROFITABILITY"] = [generate_customer_profitability(segment) for segment in data["CUSTOMER_SEGMENT"]]
data["CUSTOMER_LIFE_TIME_VALUE"] = [generate_customer_lifetime_value(segment) for segment in data["CUSTOMER_SEGMENT"]]
data["CUSTOMER_VALUE_TILL_DATE"] = [generate_customer_value_till_date(lifetime_value) for lifetime_value in data["CUSTOMER_LIFE_TIME_VALUE"]]
data["REVENUE_LAST_ONE_YEAR"] = [generate_revenue_last_year(segment) for segment in data["CUSTOMER_SEGMENT"]]
data["TOTAL_REVENUE_OVER_LIFETIME"] = data["CUSTOMER_LIFE_TIME_VALUE"]
data["NET_INTEREST_INCOME_LAST_ONE_YEAR"] = [generate_net_interest_income_last_year(segment) for segment in data["CUSTOMER_SEGMENT"]]
data["TOTAL_NON_INTEREST_INCOME"] = [generate_total_non_interest_income(segment) for segment in data["CUSTOMER_SEGMENT"]]
data["TOTAL_COST"] = [generate_total_cost(segment) for segment in data["CUSTOMER_SEGMENT"]]
data["TOTAL_ECONOMIC_PROFIT"] = pd.Series(data["REVENUE_LAST_ONE_YEAR"]) - pd.Series(data["TOTAL_COST"])
data["CUSTOMER_REGION"] = [generate_customer_region() for _ in range(num_customers)]
data["ACCOUNT_TYPE"] = [generate_account_type(segment) for segment in data["CUSTOMER_SEGMENT"]]
data["RECENCY_DAYS"] = [generate_recency_days(segment) for segment in data["CUSTOMER_SEGMENT"]]
data["FREQUENCY_LAST_YEAR"] = [generate_frequency_last_year(segment) for segment in data["CUSTOMER_SEGMENT"]]
data["MONETARY_VALUE"] = [generate_monetary_value(segment) for segment in data["CUSTOMER_SEGMENT"]]

# Create DataFrame
df = pd.DataFrame(data)

# Correlation Analysis
# Convert CUSTOMER_SEGMENT and CUSTOMER_REGION to numerical values for correlation calculation
df["CUSTOMER_SEGMENT_NUM"] = df["CUSTOMER_SEGMENT"].map({"Young Adults": 1, "Mid-Career Professionals": 2, "Retired Seniors": 3})
df["CUSTOMER_REGION_NUM"] = df["CUSTOMER_REGION"].map({"Urban": 1, "Suburban": 2, "Rural": 3})

# Select only the numeric columns for correlation analysis
correlation_matrix = df[[
    "AGE", "CUSTOMER_PROFITABILITY", "CUSTOMER_LIFE_TIME_VALUE", "CUSTOMER_VALUE_TILL_DATE",
    "REVENUE_LAST_ONE_YEAR", "TOTAL_REVENUE_OVER_LIFETIME", "NET_INTEREST_INCOME_LAST_ONE_YEAR",
    "TOTAL_NON_INTEREST_INCOME", "TOTAL_COST", "TOTAL_ECONOMIC_PROFIT", "RECENCY_DAYS",
    "FREQUENCY_LAST_YEAR", "MONETARY_VALUE", "CUSTOMER_SEGMENT_NUM", "CUSTOMER_REGION_NUM"
]].corr()

# Plotting Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Matrix of Customer Data")
plt.show()

# Display the first 10 rows of the generated data
print(df.head(10))

# Save to CSV
#df.to_csv("bank_customer_dummy_data_with_correlation.csv", index=False)
