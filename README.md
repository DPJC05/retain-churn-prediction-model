# _Retain: A Churn Prediction and Retention-Triage Tool for E-Commerce_

#### Video Demo: https://youtu.be/B95QQGt3MI0

## Project Summary

_Retain_ is a customer churn prediction and retention-triage tool designed using the UCI Online Retail II e-commerce dataset. It ingests 2 years of transactional data to generate churn probabilities for active customers as per the latest logged transaction date. These predictions are delivered via an interactive dashboard designed to help retention managers develop optimised retention campaign strategies.

## Background: Business Rationale for Churn Prediction Models

### Defining Churn

Customer churn occurs when a customer stops paying a business for its products or services. It can occur in two business settings: contractual and non-contractual. In contractual settings, customers are related to businesses via subscriptions or recurring service contracts. Here, churn is observable via subscription and contract cancellations. Conversely, in non-contractual settings (e.g., e-commerce), churn is not observable as customers simply stop returning to the business without notice. Nevertheless, non-contractual churn can still be determined statistically.

### Acquisition vs Retention

It is often estimated that acquiring a new customer costs 5-25 times more than retaining an existing one. Whilst this figure is anecdotal, it points in a sensible direction. A business already has an existing customer's attention, contact details and purchase history. The marginal cost of sending a retention email is near zero; acquiring a new customer, on the other hand, is a paid ad auction. Logically, retention should be prioritised over acquisition where possible.

The value of retention is further exemplified by a key principle in contractual churn business arithmetics: the relationship between customer lifetime value ("CLV") and churn is non-linear and inversely proportional.

In contractual settings, CLV is defined as:
```
Customer Lifetime Value ("CLV") = Average Revenue Per User ("ARPU") × Gross Margin × (1 / Churn Rate)
```
If ARPU is $50.00, Gross Margin is 70% and the monthly Churn Rate is 5%, CLV is $700.00:
```
CLV = 50.00 × 0.70 × (1 / 0.05) = 700.00
```
If the monthly Churn Rate drops to 4%, CLV grows to $875.00:
```
CLV = 50.00 × 0.70 × (1 / 0.04) = 875.00
```

This showcases that even a 1% reduction in the churn rate can raise the CLV by 25%, which can significantly boost overall business profitability. Whilst the same formula cannot be used to directly calculate CLV for non-contractual churn (due to churn being a probabilistic estimate rather than observable event), it is still reflective of the general non-linearity between CLV and churn present in non-contractual settings. This justifies effective retention management as a non-negotiable business asset.

## Churn Prediction: Triage for Optimising Finite Retention Resource Allocation

Retention teams are, however, bound by finite resources available for intervention. A churn prediction model serves to optimise resource allocation by acting as a triage tool that draws attention to customers with high churn risk. _Retain_ expands on this functionality by enabling retention managers to: (1) sort customers based on specific feature data; (2) review individual customers' feature data relative to the customer base, to inform segmented and/or targeted intervention strategies; and (3) simulate the economics of a retention campaign using adjustable parameters to determine an optimal retention campaign strategy.

## Project Structure

```
retain-churn-prediction-model/
│
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml                        # Package configuration for universal access in project folder
├── config.yaml                           # Project-level constants
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.xlsx         # Original raw UCI Online Retail II dataset
│   ├── interim/
│   │   ├── transactions.parquet          # Prepared dataset
│   │   └── data_prep_purge_report.json   # Record of number of rows dropped by src.data_prep
│   └── processed/
│       └── features.parquet              # Dataset with feature engineering
│
├── models/
│   ├── churn_pipeline.joblib             # Fitted scikit-learn Pipeline
│   ├── metadata.json                     # Record of training date, metrics, feature list, threshold, versions
│   ├── evaluation_report.json            # Record of evaluation metrics and threshold selection
│   ├── evaluation_arrays.npz             # y_test and y_proba np.ndarrays for the campaign economics simulator
│   └── curve_data.json                   # Chart.js curve data
│
├── database/
│   └── retain.db                         # SQLite database for Flask application
│
├── notebooks/
│   ├── 01_raw_eda.ipynb                  # EDA for raw UCI Online Retail II dataset
│   ├── 02_processed_eda.ipynb            # EDA for processed features.parquet dataset
│   └── 03_model_experiments.ipynb        # Prototypes for model training architecture in src.train
│
├── src/
│   ├── __init__.py
│   ├── config.py                         # Loads and validates config.yaml
│   ├── data_prep.py                      # Converts raw to interim data: executes data cleaning and preparation
│   ├── features.py                       # Converts interim to processed data: executes all feature engineering
│   ├── labelling.py                      # Constructs churn labels in processed data
│   ├── train.py                          # Builds, trains, tunes and persists a scikit-learn Pipeline
│   ├── economics.py                      # Selects a contact threshold that maximises the campaign's expected value
│   ├── evaluate.py                       # Generates evaluative metrics for model performance
│   ├── curve.py                          # Generates curve data for Chart.js
│   ├── predict.py                        # Predicts churn probability for active customers and writes scores to database
│   └── db.py                             # SQLite database functions
│
├── app/
│   ├── __init__.py                       # Flask application factory
│   ├── routes.py                         # Renders HTML templates to the browser
│   ├── api.py                            # Generates JSON endpoints for the campaign economics simulator
│   ├── helpers.py                        # Helper functions for app.routes and app.api
│   ├── static/
│   │   ├── style.css                     # Project global CSS stylesheet
│   │   ├── index.js                      # Chart.js script for risk band distribution doughnut chart
│   │   ├── performance.js                # Chart.js script for model performance evaluation curves
│   │   └── economics.js                  # Chart.js script for campaign economics sliders and expected value curve
│   └── templates/
│       ├── base.html                     # HTML dashboard webpage template containing the navigation bar
│       ├── index.html                    # Overview page with KPIs
│       ├── customers.html                # Page containing a sortable, filterable, paginated customer risk table
│       ├── customer_details.html         # Page containing details for a single-customer
│       ├── performance.html              # Page containing model evaluation curves
│       ├── economics.html                # Page containing the retention campaign economics simulator
│       ├── 400.html                      # Error 400 page
│       ├── 404.html                      # Error 404 page
│       └── 500.html                      # Error 500 page
│
├── tests/
│   ├── __init__.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── test_data_prep.py
│   │   ├── test_features.py
│   │   ├── test_labelling.py
│   │   ├── test_economics.py
│   │   └── test_predict.py
│   └── app/
│       ├── __init__.py
│       └── test_helpers.py
│
└── scripts/
    ├── __init__.py
    └── run_pipeline.py                   # Engine for single-command repository set-up
```

## Repository Download & IDE Setup

**Python Interpreter Version Requirement:** 3.13+

Execute the following commands in the terminal to obtain a local copy of the repository:

**Step 1: Clone the repository from GitHub**
```
git clone https://github.com/DPJC05/retain-churn-prediction-model.git
```

**Step 2: Move into the root directory**
```
cd retain-churn-prediction-model
```

**Step 3: Initialise the Python virtual environment**
```
python -m venv .venv
```

**Step 4: Activate the virtual environment**
```
# For MacOS/Linux (Bash/Zsh):
source .venv/bin/activate

# For Windows (Command Prompt):
.venv\Scripts\activate.bat

# For Windows (Powershell):
.\.venv\Scripts\Activate.ps1
```

**Step 5: Download the virtual environment dependencies**
```
pip install -r requirements.txt
```

**Step 6: Acquire and upload the dataset**

Download the UCI Online Retail II dataset from https://archive.ics.uci.edu/dataset/502/online+retail+ii. The dataset is to be uploaded to the ```data/raw/``` directory as an excel spreadsheet, with the file name ("online_retail_II.xlsx") left unchanged.

**Step 7: Run ```scripts.run_pipeline``` to complete the backend setup**
```
python -m scripts.run_pipeline
```

**Step 8: Open the Flask application**
```
flask --app app run
```

**Additional Notes**

1. To try a model constructed using a different set-up, edit the project-level constants in ```config.yaml``` and re-run step 7. 

2. To peruse the EDA and Model Experiments notebooks, ensure that the IDE / Code Editor used to open the repository has the relevant extensions for Jupyter Notebooks configured. Set the ipynb kernel to ```.venv``` to run the code.

## Project Functionality

### Overview
```
data/raw/online_retail_II.xlsx            1,067,371 line items
   │
   │  src/data_prep.py              cleans data, coerces data types, flags cancellations, calculates line total
   ▼
data/interim/transactions.parquet         794,244 line items
   │
   │  src/features.py               engineers dataset with one row per customer and cutoff date pair (22 features per row)
   │  src/labelling.py              constructs and appends churn labels to feature dataset
   ▼
data/processed/features.parquet           17,857 labelled rows across 6 cutoffs
   │
   │  src/train.py                  constructs, trains and compares 3 scikit-learn models, then tunes the winner
   │  src/economics.py              selects a contact threshold that maximises the campaign's expected value
   │  src/evaluate.py               evaluates model baselines and generates model evaluation metrics
   │  src/curve.py                  generates ROC, PR, cumulative gains and calibration curve data for Chart.js
   ▼
models/                                   fitted scikit-learn Pipeline + JSON reports + test arrays (y_test, y_proba)
   │
   │  src/predict.py                scores every customer active at the last date in the raw dataset based on churn risk
   │  src/db.py                     write scores to a SQLite database
   ▼
database/retain.db                        3,289 scored customers
   │
   │  app/                          Flask + Jinja + Chart.js
   ▼
Retain Churn Prediction & Retention-Triage Dashboard: Overview · Customers · Customer Details · Performance · Economics
```

As illustrated above, the executable project files are split into two packages: ```src/``` (the model pipeline) and ```app/``` (the dashboard). Akin to a main function for ```src/```, ```scripts.run_pipeline``` executes the entire model pipeline. It was implemented to enable local repository set-ups using a single terminal command.

### Project Configuration

```pyproject.toml``` ensures that all modules in the ```src/``` and ```app/``` packages can be imported by any Python script in the project folder, regardless of their location relative to the project root. 

All tunable project-level constants (the file paths, churn horizon, cutoff dates, default campaign parameters, and random seed) are stored in ```config.yaml```; they are then configured into the Python modules via the ```load_config()``` function in ```src/config.py```. This was implemented to avoid the hard-coding of the project-level constants, to enable different project set-ups with a one-line edit and re-run.

## The Dataset and Data Preparation

### UCI Online Retail II

UCI Online Retail II is an e-commerce dataset containing all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011. It contains 1,067,371 line item purchase entries distributed across 2 sheets ("2009-2010" and "2010-2011"). The business in this dataset sells unique all-occasion gift-ware. Many customers of the company are wholesalers. The raw dataset schema is as follows:

| Column | Description |
|--------|-------------|
| Invoice | Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'C', it indicates a cancellation. |
| StockCode | Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product. |
| Description | Product (item) name. Nominal. |
| Quantity | The quantities of each product (item) per transaction. Numeric. |
| InvoiceDate | Invice date and time. Numeric. The day and time when a transaction was generated. |
| Price | Unit price. Numeric. Product price per unit in sterling (Â£). |
| Customer ID | Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer. |
| Country | Country name. Nominal. The name of the country where a customer resides. |
<p></p>

### Data Preparation

```src/data_prep.py``` converts the raw UCI Online Retail II dataset into a cleaned transaction table. ```src/data_prep.py``` functions as follows:

1. ```data/raw/online_retail_II.xlsx``` is imported as 2 separate pd.DataFrame objects (1 per sheet) and concatenated.

2. All column headers are reformatted to snake_case. This eliminates the space in "Customer ID" to enable direct indexing into the pd.DataFrame, and aligns the dataset with Python naming conventions.

3. The following rows are dropped, with the number of rows dropped at each stage recorded in a JSON purge report.

| Step | Rows Dropped | Explanation |
|------|-------------:|-------------|
| Drop exact duplicate rows | 34,335 | Duplicated line items would inflate spend and order counts. |
| Drop rows with no customer ID | 235,151 | Guest transactions cannot be tracked over time, so they cannot be used for customer-level modelling. All rows with a missing description also had a missing Customer ID, so this step simultaneously removed every missing value in the raw data. |
| Drop non-product stock codes | 3,641 | Codes such as `POST` (postage), `M` (manual), `D` (discount), `BANK CHARGES`, `ADJUST` and `TEST001` were identified during the exploratory data analysis to be accounting entries rather than purchases. They were also found to carried extreme prices (up to £38,970) that would distort spending features. |
| **Rows Remaining:** | **794,244** | |
<p></p>

4. Each set of column data is coerced to an explicit data type:

| Column | Data Type |
|--------|-----------|
| invoice | pd.string |
| stock_code | pd.string |
| description | pd.string |
| quantity | np.int32 |
| invoice_date | pd.datetime64[us] |
| price | np.float64 |
| customer_id | np.int64 |
| country | pd.string |
<p></p>

5. A ```cancelled``` flag is added to all the rows where the invoices are prepended with "C".

6. ```line_total``` is calcuated for and appended to all rows using the formula ```line_total = price × quantity```.

7. The interim dataset is persisted as ```data/interim/transactions.parquet```. The parquet file type (.parquet) was chosen for its: (a) small file size; (b) fast import time as compared to Excel files (.xlsx); and (c) ability to retain data types in storage. 

8. The purge report is persisted as ```data/interim/data_prep_purge_report.json```. Whilst building the project, this purge report was verified against the purge counts in ```01_raw_eda.ipynb``` to ensure correct implementation.

## Defining Churn

### The Observation and Outcome Windows

Each row in the feature dataset is built around a cutoff date _T_. Refer to the timeline below:

```
|<-------- OBSERVATION WINDOW -------->|<------ OUTCOME WINDOW ------>|
^                                      ^                              ^
First Transaction               Cutoff Date (T)                T + Horizon (h)
```

Features are engineered using data from the observation window only. Labels are constructed using data from the outcome window only. When predicting churn, the scikit-learn model is not exposed to any data data from the outcome window except for the 'churned' (0 or 1) label. A customer is labelled as churned if they fail to make a new transaction (excluding cancellations) within _h_ days after the cutoff date. 

### Selecting the Churn Horizon (_h_)

The churn horizon _h_ drives the model's base rate. If set too low, less time is available for customers to make a purchase, which may result in them being prematurely labelled as churned. If set too high, genuine churns may be unintentionally labelled as active. Moreover, a high churn horizon value would also cost training data, as every observation window cutoff requires a full outcome window after it. Careful churn horizon value selection is therefore essential to effective non-contractual churn modelling. 

Moveover, all customers will churn eventually. Building a model using a high churn horizon value would limit its efficacy, as predictions that look a year ahead (or similar) offer limited actionable intervention opportunities. The horizon should create an outcome window that is short enough to act on, yet long enough to accurately predict churn.

For this project _h_ was derived from the data in ```notebooks/01_raw_eda.ipynb``` via a review of the intervals between consecutive orders:

- **Pooling every interval between consecutive purchases in a distribution revealed that 90% of purchases would be followed by another purchase within 135 days**. However, this figure understated the true horizon, as frequency weighting lowered the quantile values. Hence, 135 days was identified as a minimum baseline value, rather than a churn horizon candidate. 

- Aggregating intervals to one value per customer removed the impact of frequency weighting. **The mean and median interval distributions agreed that 80% of customers would return to make a purchase within 143-151 days**. Their 90th and 95th percentiles were also congruent, but sat at 222-223 and 323-324 days, which significantly limited the amount of usable training data. The maximum interval duration distribution presented similar signals, with a median of 151 days and higher quantiles running into the same constraint.

- Based on these results, **150 days was adopted for _h_**, rounded from the 143-151 days range.

### Stacking Multiple Snapshots (Built From Different Cutoff Dates)

UCI Online Retail II only contains 5,876 identified customers with real purchases. This is a limited quantity for model training and testing. To alleviate this, a multi-snapshot design was adopted. This involved stacking multiple snapshots built on different cutoffs to maximise the contribution of each customer to the feature dataset (as each customer would appear in multiple datasets, with different features based on the cutoff date). Refer to the figure below:

```
Snapshot 1:  |<--- OBSERVATION WINDOW --->|<---- OUTCOME WINDOW ---->|--------------------------------------------|
(Training)   ^                            ^                          ^                                            ^
             01/12/2009                   2010-10-01                 2010-10-01 + h (2011-02-28)                  09/12/2011

Snapshot 2:  |<--- OBSERVATION WINDOW ------>|<---- OUTCOME WINDOW ---->|-----------------------------------------|
(Training)   ^                               ^                          ^                                         ^
             01/12/2009                      2010-11-01                 2010-11-01 + h (2011-03-31)               09/12/2011

Snapshot 3:  |<--- OBSERVATION WINDOW --------->|<---- OUTCOME WINDOW ---->|--------------------------------------|
(Training)   ^                                  ^                          ^                                      ^
             01/12/2009                         2010-12-01                 2010-12-01 + h (2011-04-30)            09/12/2011

Snapshot 4:  |<--- OBSERVATION WINDOW ------------>|<---- OUTCOME WINDOW ---->|-----------------------------------|
(Training)   ^                                     ^                          ^                                   ^
             01/12/2009                            2011-01-01                 2011-01-01 + h (2011-05-31)         09/12/2011

Snapshot 5:  |<--- OBSERVATION WINDOW ---------------->|<---- OUTCOME WINDOW ---->|-------------------------------|
(Training)   ^                                         ^                          ^                               ^
             01/12/2009                                2011-02-01                 2011-02-01 + h (2011-07-01)     09/12/2011

Snapshot 6:  |<--- OBSERVATION WINDOW ------------------------------------------------>|<---- OUTCOME WINDOW ---->|
(Testing)    ^                                                                         ^                          ^
             01/12/2009                                                                2011-07-12                 09/12/2011
```

The test dataset (snapshot 6) cutoff date, 2011-07-12, was selected as the last date with a full 150-day outcome window before the data ends on 2011-12-09. To ensure that no data from the test dataset would be seen by the scikit-learn model during training, the 150 days before 2011-07-12 was implemented as an embargo period, which left 2011-02-12 as a 2nd-latest cutoff date. 2011-02-01 was selected as the last training dataset cutoff date (snapshot 5) for easier window tracking. The cutoff dates were then decremented monthly until 2010-10-01 to create 5 snapshots for model training, with at least 10 months of transaction history before snapshot 1.

Note that each snapshot only contained customers who made a new order (cancellations excluded) within _h_ days before _T_. This eligibility condition was included to filter out inactive customers who had already churned before the cutoff date, to ensure that the features were reflective of active customers only and therefore valid signals for predicting churn.

Ultimately, this resulted in the final dataset containing 17,857 labelled rows: 15,311 for training (3,836 distinct customers) and 2,546 for testing. Refer to the table below for a breakdown of the eligible customer counts and churn rates across the 6 snapshots.

| Cutoff | Role | Eligible Customers | Churn Rate |
|--------|------|-------------------:|-----------:|
| 2010-10-01 | Training | 2,669 | 35.3% |
| 2010-11-01 | Training | 2,982 | 41.0% |
| 2010-12-01 | Training | 3,283 | 49.0% |
| 2011-01-01 | Training | 3,242 | 49.6% |
| 2011-02-01 | Training | 3,135 | 47.4% |
| 2011-07-12 | Test | 2,546 | 30.4% |

## Feature Engineering & Label Construction

```src/features.py``` collapses each customer's observation-window transactions into a single row of engineered feature data. To avoid data leakage, every function that builds features takes the cutoff date as an argument and filters the transactions using this date as its first step, to ensure that no features contain data from after the cutoff. Cancelled invoices are excluded from all features except the return rate. 

The table below outlines the 22 features engineered for each customer/cutoff pair:

| S/N | Feature | Description | Units | Data Type |
|-----|---------|-------------|-------|-----------|
| - | customer_id | Unique 5-digit customer identification number (dropped before model fitting). | - | np.int64 |
| - | cutoff | Cutoff date for observation window (dropped before model fitting). | - | pd.datetime64[us] |
| - | churn_horizon | Number of days leading to the churn horizon (dropped before model fitting). | Days | np.int64 |
| 1 | recency | Number of days since last purchase. | Days | np.int64 |
| 2 | frequency | Number of purchases since first purchase. | EA | np.int64 |
| 3 | monetary | Gross expenditure since first purchase. | Sterling (Â£) | np.float64 |
| 4 | mean_order_value | Average gross expenditure per order. | Sterling (Â£) | np.float64 |
| 5 | tenure | Number of days since first purchase (time spent as a registered customer). | Days | np.int64 |
| 6 | avg_inter_purchase_gap | Average number of days between orders. | Days | np.float64 |
| 7 | std_inter_purchase_gap | Standard deviation in the number of days between orders. | Days | np.float64 |
| 8 | recency_ratio | Ratio representing recency as compared to the average number of days between orders. | - | np.float64 |
| 9 | spend_30d | Gross expenditure within 30 days before cutoff. | Sterling (Â£) | np.float64 |
| 10 | spend_90d | Gross expenditure within 90 days before cutoff. | Sterling (Â£) | np.float64 |
| 11 | spend_150d | Gross expenditure within 150 days before cutoff. | Sterling (Â£) | np.float64 |
| 12 | orders_30d | Number of orders within 30 days before cutoff. | EA | np.float64 |
| 13 | orders_90d | Number of orders within 90 days before cutoff. | EA | np.float64 |
| 14 | orders_150d | Number of orders within 150 days before cutoff. | EA | np.float64 |
| 15 | spend_momentum_30d_90d | Ratio representing gross expenditure within 30 days as compared to 90 days before cutoff. | - | np.float64 |
| 16 | spend_momentum_90d_150d | Ratio representing gross expenditure within 90 days as compared to 150 days before cutoff. | - | np.float64 |
| 17 | order_momentum_30d_90d | Ratio representing number of orders within 30 days as compared to 90 days before cutoff. | - | np.float64 |
| 18 | order_momentum_90d_150d | Ratio representing number of orders within 90 days as compared to 150 days before cutoff. | - | np.float64 |
| 19 | distinct_products | Number of products purchased with unique stock codes since first purchase. | EA | np.int64 |
| 20 | mean_order_quantity | Average quantity per product purchased per order. | EA | np.float64 |
| 21 | return_rate | Proportion of total number of orders that represents orders returned. | - | np.float64 |
| 22 | country | Modal country in which orders were placed. | - | pd.category |
<p></p>

The following features contained missing values post-feature engineering. These were deliberately left as ```np.NaN``` in the persisted feature dataset, as their missing status was informative, and they could be imputed within the model pipeline.

| Feature | Number of Rows | Affected Customers |
|---------|---------------:|--------------------|
| avg_inter_purchase_gap | 4,082 | Customers with only 1 order in their observation window. |
| std_inter_purchase_gap | 7,225 | Customers with 2 or less orders in their observation window. |
| recency_ratio | 4,281 | Customers with only 1 purchase in their observation window **OR** with all their purchases on the same day. |
| spend_momentum_30d_90d | 3,705 | Customers who had no purchase activities within 90 days before the cutoff date. |
| order_momentum_30d_90d | 3,705 | Customers who did not make any orders within 90 days before the cutoff date. |
<p></p>

```src/labelling.py``` subsequently constructs the labels using each customer's outcome-window transaction data, and merges the labels onto the existing feature dataset on the (customer_id, cutoff) pair. 

## Model Training and Evaluation

All model training is completed by ```src/train.py```, with the assistance of functions imported from ```src/economics.py```, ```src/evaluate.py``` and ```src/curve.py```.

### Temporal Train-Test Split

As the feature dataset is built on chronologically-ordered snapshots, a random 80-20 split would result in data leakage, with the scikit-learn model making predictions for customers in snapshots 1-5 after being trained on customers in snapshot 6. Hence, a temporal split approach was adopted for this project, with snapshots 1-5 constituting the training set (15,311 rows) and snapshot 6 constituting the testing set (2,546 rows).

#### Note on the Original Design

```temporal_split()``` originally purged customers that appeared in both the training and testing sets to avoid potential data leakage. However, when the function was tested in ```notebooks/03_model_experiments.ipynb```, concerning metrics were found in the purged report. Firstly, 1827 customers were purged; this was almost half of the training set's customer count. Secondly, the training set churned rate rose to 80.09%. These metrics suggested that customers who appeared in both sets made up the majority of customers who did not churn in the training set. Purging the data would ultimately result in the model overpredicting churn, due to the significant churn rate disparity between both datasets. 

Given that the customer/cutoff pair would be dropped when building X_train and X_test, and the different cutoffs would result in different feature values (despite the overlapping activity history), I recognised that the overlapping customers in the training and testing sets would ultimately be treated as different entries by the model while fitting, with no actual data leakage occurring. As such, the purging element of ```temporal_split()``` was removed.

### Preprocessing

Every candidate model is a full scikit-learn ```Pipeline```, with a ```ColumnTransformer``` routing each column through one of three branches:

1. **Numerical** (recency, tenure, the momentum ratios, return_rate). The dataset undergoes median imputation with a missing-value indicator, followed by standard scaling.

2. **Skewed Numerical** (frequency, monetary, mean_order_value, the interval features, recency_ratio, distinct_products, mean_order_quantity). The dataset undergoes median imputation with an indicator, followed by ```log1p``` transformation and standard scaling. ```notebooks/02_processed_eda.ipynb``` identified the columns with skewed data by selecting the features with maximum values that were signficantly higher than their medians.

3. **Categorical** (country). The dataset undergoes most-frequent imputation, followed by one-hot encoding with any country below 1% of rows categorised into an "infrequent" bucket. Under this system, only the United Kingdom, Germany and France are identified. ```notebooks/02_processed_eda.ipynb``` showed that German and French customers had a noticeably lower churn rate as compared to other countries (28% and 23% against roughly 43-46%), therefore providing unique signals that earned their own columns.

### Model Candidate Construction, Comparison and Tuning

Three classifier models (Logistic Regression, Random Forest and Gradient Boosting) are compared on identical data with five-fold ```StratifiedGroupKFold``` cross-validation. Grouping by Customer ID keeps all of a customer's rows in the same fold, so the model is never validated on a customer it has just been trained on. A winning candidate is then selected based on the ROC-AUC score, and tuned with ```GridSearchCV``` over the same grouped folds, scoring on ROC-AUC, PR-AUC and Brier score and refitting on ROC-AUC. The tuned winner is then persisted as ```models/churn_pipeline.joblib```.

### Model Evaluation

```src/train.py``` then calls helper functions from ```src/evaluation.py``` and ```src/curve.py``` to generate the evaluation metrics and curve data respectively. These are persisted alongside the tuned winner as model metadata JSON files.

## Campaign Economics

Whilst churn predictions draw attention to customers with higher churn risk, they do not instruct retention managers on which customers to contact or indicate if a retention campaign will be profitable. ```src/economics.py``` and the campaign economics simulator in the dashboard serve to bridge this gap and transform churn predictions into actionable business decisions.

### Contact Thresholds and Campaign Expected Values

A contact threshold is the minimum churn probability required for a customer to remain on the contact list. All customers with a churn probability equal to or higher than the contact threshold will be contacted. 

The expected value of a retention campaign is determined via the following formula, which measures the value added by the retention campaign as compared to zero intervention:

```
Expected Value = (Number of Successful Interventions × Customer Lifetime Value) - Intervention Cost
```

which translates to:

```
Expected Value = (True Positives × Save Rate × Customer Lifetime Value) − (Customers Contacted × Cost per Contact)
```

Each contact threshold has a different expected value. The ```select_threshold()``` function from ```src/economics.py``` serves to determine the contact threshold that would enable the greatest possible expected value, based on capacity constraints.

### Scaling Capacity

When retention managers determine their campaign capacity, the decision is made relative to the live customer count. However, as the campaign economics simulator is computed counterfactually based on the test set, which may be a different size. ```scale_capacity()``` was implemented maintain the proportion of live customer count constituted by the defined campaign capacity.

### Threshold Selection

```select_threshold()``` computes the expected value for 99 contact thresholds (0.01 to 0.99) and selects the threshold with the highest value. This expected value is then checked against the campaign capacity constraint. If the selected contact threshold contains a customer count greater than the campaign capacity constraint, the capacity is found to be **binding**. Here, the campaign expected value is recalculated to reflect contacting the top _N_ (where _N_ refers to the scaled capacity) customers with the highest churn risk. An additional forgone value is also calculated to indicate the loss incurred by the constrained capacity.

### Default Campaign Parameters

To determine an initial contact threshold and expected, the following default campaign parameters were implemented in ```config.yaml```:

| Campaign Parameter | Default | Origin |
|--------------------|--------:|--------|
| Cost per Contact | £2.50 | A value estimating the intervention cost. The actual value is dependent on the methods of intervention undertaken by retention teams. |
| Customer Lifetime Value | £451.50 | The median expenditure by active customers over a 150-day window, determined via the exploratory data analysis. As the mean expenditure per customer is higher due to wholesalers (£1,096), the median expenditure is a fairer figure for a typical customer. |
| Save Rate | 30% | An value estimating the save rate. As the raw dataset does not contain any interventions, it is not possible to compute the save rate statistically. |
| Campaign Capacity | 500 | A placeholder of roughly 20% of the test population. The actual value is dependent on the manpower and resource constraints in retention teams. |
<p></p>

Capacity binds heavily under the default campaign parameters. Due to the Cost per Contact being cheap relative to Customer Lifetime Value, the unconstrained profit-maximising threshold (listed in ```models/evaluation_report.json```) is 0.02, which instructs to contact 2,475 of the 2,546 test customers for an expected value of £98,786. Limited to 500 contacts, the default selected contact threshold rises to 0.638, and the campaign is worth £41,010, so the capacity limit costs £57,776. This contact threshold of 0.638 was used to assign the risk bands in ```src/predict.py```.

Additionally, to account for the save rate being an estimate, the following is also computed:

1. A **sensitivity grid**: the expected value at five save rates from 20% to 40% (from £26,924 to £55,097 at the chosen threshold).

2. A **break-even save rate**: the save rate at which the campaign stops paying for itself. In this project, it is under 1%, which indicates that the retention campaign would be profitable regardless of the save rate estimate.

3. A **break-even probability**: the minimum churn probability where, on average, contacting a single customer would pay for itself (cost ÷ (save rate × lifetime value), about 0.018).

## Churn Prediction and Risk Band Allocation

```src/predict.py``` rebuilds a feature dataset for all the eligible customers as of the latest date in the transaction log, scores them with churn probabilities using the persisted scikit-learn Pipeline, and assigns them with their corresponding churn risk band. It then uses functions from ```src/db.py``` to write the prediction data to the ```customer_scores``` table in ```database/retain.db```. Refer to the ```customer_scores``` table schema below:

```
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customer_scores (
    customer_id INTEGER PRIMARY KEY,
    churn_proba REAL NOT NULL,
    risk_band TEXT NOT NULL,
    recency INTEGER,
    frequency INTEGER,
    monetary REAL,
    tenure INTEGER,
    country TEXT,
    predicted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scores_proba ON customer_scores(churn_proba DESC);
CREATE INDEX IF NOT EXISTS idx_scores_band ON customer_scores(risk_band);
"""
```

Each run of ```scripts/run_pipeline.py``` clears and rewrites the ```customer_scores``` table. This ensures that ```database/retain.db``` always reflects a single churn probability scoring pass.

## The Dashboard

The Flask app in ```app/``` is built using an application factory and two blueprints: one for rendering the HTML pages and another for the campaign economics simulator JSON API. Upon starting the app via ```flask --app app run```, the ```create_app()``` function loads the model metadata, evaluation report, curve data and test arrays into memory. The app will not start if any of the model artifacts are missing, and will issue a warning message if they are from different training runs. Refer to the table below for all of the routes implemented:

| Route | Method | Page |
|-------|--------|------|
| ```/``` | GET | **Overview.** Contains the headline figures (customers scored, mean churn risk, number of high-risk customers, and the revenue at risk), a doughnut chart showing the risk band distribution, and a panel containing the provenance of the underlying model. |
| ```/customers``` | GET | **Customers.** Contains a paginated, filterable and sortable table of scored customers. |
| ```/customers/export``` | GET | Exports and downloads the filtered customer table as a CSV file. |
| ```/customers/<id>``` | GET | **Customer Details.** Contains a customer's individual feature profile with percentiles relative to the customer base. |
| ```/performance``` | GET | **Performance.** Contains the headline evaluation metrics (ROC-AUC, PR-AUC, base rate, Brier score, and the gap between cross-validation and test ROC-AUC scores); the ROC, precision-recall, cumulative gains and calibration curves; a complete evaluation table (including the model baselines); and a table containing lift by decile. |
| ```/economics``` | GET | **Economics.** Contains the campaign simulator: 4 sliders (cost per contact, customer lifetime value, save rate, and capacity); the expected value curve with a shaded band for the save rate (±0.10) as well as markers for the chosen and unconstrained thresholds; the outcome metrics (expected value, targeted capacity, precision and recall); a banner for when capacity is binding; and the sensitivity grid table. |
| ```/api/economics``` | POST | Reads the 4 adjusted campaign parameters as JSON and returns the recalculated campaign data. To enable live updates, ```app/static/economics.js``` throttles slider input to one request every 100 milliseconds, ignoring any response that arrives after each new request is sent, and updates the expected value curve in place rather than deleting and redrawing it. |

## Unit Testing Regime

The unit tests cover 35 functions across 6 modules. Refer to the table below for an overview of the testing regime:

| Unit Test Script | Module Tested | Functionality Tested |
|------------------|---------------|----------------------|
| ```tests/src/test_data_prep.py``` | ```src/data_prep.py``` | Column renaming; duplicate, missing customer_id, and non-product dropping and drop count recording; data type coercion; cancellation flagging; and line_total calculations. |
| ```tests/src/test_features.py``` | ```src/features.py``` | Window filtering; eligibility filting; accurate feature construction of all features. |
| ```tests/src/test_labelling.py``` | ```src/labelling.py``` | Eligibility filtering; outcome window filtering; accurate label construction. |
| ```tests/src/test_economics.py``` | ```src/economics.py``` | Capacity scaling; confusion count calculations; expected value calculations; top-_N_ campaign calculations; capacity binding determination and handling; the basic property that increasing the save rate should never lower the expected value; and break-even save rate calculations. |
| ```tests/src/test_predict.py``` | ```src/predict.py``` | Accurate risk band assignment. |
| ```tests/app/test_helpers.py``` | ```app/helpers.py``` | Customer table parsing; customer table pagination building; population percentile comparisons; and accurate rejections of invalid economics parameters. |

## Data Source

Chen, D. (2012). _Online Retail II_ [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
