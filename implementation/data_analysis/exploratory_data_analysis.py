import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, MaxNLocator
import seaborn as sns
import json
import datetime
import logging
from typing import Callable


## Adjusting logging

error_logger: logging.Logger = logging.getLogger("debug_logger")
error_logger.setLevel(logging.ERROR)
error_handler: logging.FileHandler = logging.FileHandler(
    filename="error_logger.log", mode="w"
)
error_formatter = logging.Formatter(
    fmt="%(name)s %(asctime)s %(message)s\nLine: %(lineno)s"
)
error_handler.setFormatter(fmt=error_formatter)
error_logger.addHandler(hdlr=error_handler)

debug_logger: logging.Logger = logging.getLogger("debug_logger")
debug_logger.setLevel(logging.DEBUG)
debug_handler: logging.FileHandler = logging.FileHandler(
    filename="debug_logger.log", mode="w"
)
debug_formatter = logging.Formatter(fmt="%(name)s %(asctime)s %(message)s\nLine: %(lineno)s")
debug_handler.setFormatter(fmt=debug_formatter)
debug_logger.addHandler(hdlr=debug_handler)


### EDA

image_save_dir: str = "assets\\visualizations\\"

with open(file=(
    "C:\\Users\\artjo\\.vscode\\Space StartUps\\assets\\data\\startups_data.json"
    ),
    mode="r",
    encoding="utf-8"
) as file:
    data: dict[str, str] = json.load(fp=file)
    
data: pd.DataFrame = pd.DataFrame(data=data)

# While data scrapping not founded values were recorded as "Unknown"
data = data.replace(to_replace="Unknown", value=None)

print(data.info())  # =>
"""
- Data has 130 rows and 7 columns.
- All the columns have 130 non-null objects, therefore there are
    no missing values in data.
"""

print(data.nunique()) # =>
"""
Name and Idea columns have all the 130 values unique.
Data contains:
1. 21 locations (countries) - categorical data
2. 20 unique year values - categorical data
3. 6 employees number categories - categorical data
4. 5 funding levels - categorical data
5. 70 unique amount raised values - numerical data
....
"""

print(data.columns) # =>
"""
All columns are renamed for dataframe readability.
Also, columns Location, Founded, Idea are concretized.
"""
remove_non_ch: Callable = lambda ch: ch if ch.isascii() or ch == " " else ""
increase_readability: Callable = lambda col_name:(
    "_".join(
        (
            "".join(
                map(remove_non_ch, col_name)
            )
        ).split()
    ).lower()
)

readable_column_names: list[str] = (
    list(map(increase_readability, data.columns))
)
change_names_dict: dict[str, str] = (
    dict(zip(data.columns, readable_column_names))
)
change_names_dict["Number of employees"] = "employees_number"
change_names_dict["Location"] = "country"
change_names_dict["Founded"] = "year_founded"
change_names_dict["Idea"] = "description"

data = data.rename(columns=change_names_dict)

""" 
Column Idea is excluded from the main dataframe. It is not relevant for EDA.
"""
idea_data: pd.DataFrame = data[["name", "description"]]
main_data: pd.DataFrame = data.drop(columns="description", axis=1)

print(main_data.head())

# Changing columns' types.
main_data["year_founded"] = (
    main_data["year_founded"].apply(func=pd.to_numeric)
)
main_data["amount_raised(usd)"] = (
    main_data["amount_raised(usd)"].apply(func=pd.to_numeric)
)

# Creating new column startup_age.
main_data.insert(
    loc=3,
    column="startup_age",
    value=datetime.date.today().year - main_data["year_founded"]
)
# Creating new column startup_size.
size_names: list[str] = [
    "Very Small", "Small", "Medium", "Large", "Very Large", "Enterprise"
]
mapping: dict[str, str] = dict(
    zip(
        main_data["employees_number"].unique(), size_names
    )
)
main_data.insert(
    loc=4,
    column="startup_size",
    value=main_data["employees_number"].map(mapping)
)
print(main_data.head())


 # Checking for anomalies in values.
for column in main_data.columns:
    print(main_data[column].unique()) if not column in [
        "name", "amount_raised(usd)"
    ] else None

pd.set_option("display.float_format", "{:.0f}".format)
print(main_data.describe().T)
"""
- Space startups' foundation years - 1989-2021.
- The oldest space startup is 35 years old, the youngest - 3
- The amounts are between 0 and 3 000 000 000, averagely 71,5 millions.
    Raised amounts' standart deviation is 370 636 877, which is very big.
"""

categorical_columns: list[str] = list(
    main_data.select_dtypes(include="object").drop(columns=["name"]).columns
)
numerical_columns: list[str] = list(
    main_data.select_dtypes(include=np.number).columns
)


## Univariate analysis

amount_formatter: FuncFormatter = FuncFormatter(
    lambda amount, _: (f"{amount:,.0f}").replace(",", " ")
)
for column in numerical_columns:
    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=(15, 5), layout="constrained"
    )
    if column == "amount_raised(usd)":
        sns.histplot(data=main_data[column], ax=axs[0], bins=100, kde=True)
    else:
        sns.histplot(data=main_data[column], ax=axs[0], kde=True)
    sns.boxplot(data=main_data[column], ax=axs[1], orient="h")
    for ax in axs:
        if column == "amount_raised(usd)":
            ax.xaxis.set_minor_locator(MultipleLocator(100000000))
            ax.tick_params(axis="x", labelrotation=45)
            ax.xaxis.set_major_formatter(amount_formatter)
        else:
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.savefig(f"{image_save_dir}{column}_skew.png")
    plt.close(fig=fig)
"""
- Space startups were started to be actively developed in around 2005.
    Whereas the boom was in around 2013-2020.
- Very few space startups are more than 15 years old.
    As a rule - 5-12 years old.
- There a few outliers, which are 20-35 years old.
- Overwhelming majority of space startups have zero raised amount, 
    much less amount are close to zero - 0-200 000 000.
    There are a few outliers with about
    1 000 000 000 - 3 000 000 000 amount raised.
"""

fig, axs = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(18, 15),
    layout="constrained"
)
print(categorical_columns)
cat_column_ind: int = 0
for row in range(2):
    for column in range(2):
        column_name: str = categorical_columns[cat_column_ind]
        sns.countplot(
            ax=axs[row, column], x=main_data[column_name], palette="Paired"
        )
        axs[row, column].yaxis.set_minor_locator(MultipleLocator(1))
        axs[row, column].grid(
            axis="y",
            visible=True,
            which="minor",
            linewidth=0.1,
            color="grey",
            linestyle="--"
        )
        axs[row, column].grid(
            axis="y",
            visible=True,
            which="major",
            linewidth=0.5,
            color="black",
            linestyle="--"
            )
        cat_column_ind += 1
axs[0, 0].tick_params(axis="x", rotation=90)
plt.savefig(f"{image_save_dir}categorical_unvariate_observ.png")
plt.close(fig=fig)
"""
Categorical data countplots show us:
- USA is a leader in space startups - more than 70. 
    The second is UK - more than 10.
- As a rule, there are very small and small size space startups with
    1-10 or 10-20 employees. Also, there are more than 15 large
    space startups with 51-200 employees.
- Mainly, space startups are currently on Seed or Self-founded funding level.
    Also, the every next funding level appears fewer times,
    which is not surprising.
"""

# Normalizing amount data distribution.
# Using log1p since the data has zero values.
main_data["amount_raised_log"] = np.log1p(main_data["amount_raised(usd)"])
plt.figure(layout="constrained")
sns.histplot(data=main_data["amount_raised_log"], bins=50, kde=True)
plt.savefig(f"{image_save_dir}amount_log_distribution")
plt.close()

numerical_columns.append("amount_raised_log")


## Bivariate analysis

numerical_pairplot: sns.PairGrid = sns.pairplot(
    data=main_data.drop(columns=["amount_raised(usd)"]),
    height=3,
    aspect=1
)
numerical_pairplot.figure.set_constrained_layout(True)
plt.savefig(f"{image_save_dir}numerical_bivariate_observ.png")
plt.close()
"""
- Startup age has positive correlation with raised amount.
  Hence founding year has negative correlation with raised amount.
"""

print(categorical_columns, numerical_columns, sep="\n")

fig, axs = plt.subplots(
    nrows=4, ncols=2, layout="constrained", figsize=(18, 25)
)
data = main_data.groupby(
    "country"
)["startup_age"].mean().sort_values(ascending=False).head(20).reset_index()
sns.barplot(
    ax=axs[0, 0],
    data=data,
    x="country",
    y="startup_age",
    palette="viridis"
)
axs[0, 0].tick_params(axis="x", rotation=90)
axs[0, 0].set_title("Country VS Age")

data = main_data.groupby(
    "startup_size"
)["startup_age"].mean().sort_values(ascending=False).reset_index()
sns.barplot(
    ax=axs[0, 1],
    data=data,
    x="startup_size",
    y="startup_age",
    palette="viridis"
)
axs[0, 1].set_title("Size VS Age")

data = main_data.groupby(
    "startup_size"
)["amount_raised_log"].mean().sort_values(ascending=False).reset_index()
sns.barplot(
    ax=axs[1, 0],
    data=data,
    x="startup_size",
    y="amount_raised_log",
    palette="viridis"
)
axs[1, 0].set_title("Size VS Amount_raised(log)")

data = main_data.groupby(
    "startup_size"
)["amount_raised(usd)"].mean().sort_values(ascending=False).reset_index()
sns.barplot(
    ax=axs[1, 1],
    data=data,
    x="startup_size",
    y="amount_raised(usd)",
    palette="viridis"
)
axs[1, 1].set_title("Size VS Amount_raised(USD)")

data = main_data.groupby(
    "current_funding_level"
)["amount_raised_log"].mean().sort_values(ascending=False).reset_index()
sns.barplot(
    ax=axs[2, 0],
    data=data,
    x="current_funding_level",
    y="amount_raised_log",
    palette="viridis"
)
axs[2, 0].set_title("Funding_lvl VS Amount_raised(log)")

data = main_data.groupby(
    "current_funding_level"
)["amount_raised(usd)"].mean().sort_values(ascending=False).reset_index()
sns.barplot(
    ax=axs[2, 1],
    data=data,
    x="current_funding_level",
    y="amount_raised(usd)",
    palette="viridis"
)
axs[2, 1].set_title("Funding_lvl VS Amount_raised(USD)")

data = main_data.groupby(
    "current_funding_level"
)["startup_age"].mean().sort_values(ascending=False).reset_index()
sns.barplot(
    ax=axs[3, 0],
    data=data,
    x="current_funding_level",
    y="startup_age",
    palette="viridis"
)
axs[3, 0].set_title("Funding_lvl VS Age")
axs[3, 1].remove()

plt.savefig(f"{image_save_dir}categorical_bivariate_observ.png")
plt.close()

"""
- On average, Japan has the oldest space startups.
- On average, the oldest space startups are
    in size of enterprises (1000+ employees)
- On average, the biggest amounts are raised in enterprises(1000+ employees).
    But the next are medium size space startups.
- On average, space startups on Series C+ funding level are 
    the oldest ones and have the biggest amounts raised.
"""


## Multivariate analysis

correlation_data: pd.DataFrame = main_data
num_funding_lvl_mapping: dict[str, str] = {
    "Self-funded": 0, "Seed": 1, "Series A": 2, "Series B": 3, "Series C+": 4
}
num_size_mapping: dict[str, str] = {
    "Very Small": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "Very Large": 5,
    "Enterprise": 6
}

correlation_data.insert(
    loc=0,
    column="startup_size(num)",
    value=correlation_data["startup_size"].str.strip().map(
        num_size_mapping
    )
)
correlation_data.insert(
    loc=0,
    column="current_funding_level(num)",
    value=correlation_data["current_funding_level"].str.strip().map(
        num_funding_lvl_mapping
    )
)
correlation_data = correlation_data.drop(
    columns=[
        "name",
        "country",
        "startup_size",
        "employees_number",
        "current_funding_level",
        "amount_raised(usd)"
    ]
)


plt.figure(figsize=(12, 7), layout="constrained")
sns.heatmap(data=correlation_data.corr(), vmin=-1, vmax=1, annot=True)

plt.savefig(f"{image_save_dir}correlation_multivariate.png")
plt.close()

"""
- Current funding level has a strong positive correlation to
    startup size (0.63) and raised amount (0.7). Also, there is a
    moderate correlation to startup age positively (0.36).
- Startup size has a strong positive correlation to startup age (0.54),
    raised amount (0.59) and current funding level (0.63).
- Startup age has a moderate positive correlation to raised amount (0.41) and
    current funding level (0.36).
"""



