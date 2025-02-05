<h2>Analysis of Diabetic Population Claims Data</h2>

<div style="page-break-after: always;"></div>

## Table of Contents
- [0. Executive Summary](#0-executive-summary)
- [1. Technical Report Overview](#1-technical-report-overview)
  - [1.1 Project Aim](#11-project-aim)
  - [1.2 Objectives](#12-objectives)
  - [1.3 Skills Matrix](#13-skills-matrix)
  - [1.4 Project Overview](#14-project-overview)
- [2. Preprocessing Steps](#2-preprocessing-steps)
  - [2.1 Data Ingestion](#21-data-ingestion)
  - [2.2 Data Cleaning](#22-data-cleaning)
    - [2.2.1 Identifying Missing Data](#221-identifying-missing-data)
    - [2.2.2 Handling Missing Data](#222-handling-missing-data)
    - [2.2.3 Checking for Duplicates](#223-checking-for-duplicates)
    - [2.2.4 Data Validation](#224-data-validation)
  - [2.4 Data Quality checks](#24-data-quality-checks)
    - [2.4.1 Logical Value Checks](#241-logical-value-checks)
    - [2.4.2 Identifying Family Units Using member\_code](#242-identifying-family-units-using-member_code)
    - [2.4.3 Correct code mapping](#243-correct-code-mapping)
    - [2.3.1 Issues with Claim Type into an Indicator](#231-issues-with-claim-type-into-an-indicator)
    - [2.3.2. Non-Standardized City Names](#232-non-standardized-city-names)
    - [2.4.4 Intermediate Quality Report \& Documentation](#244-intermediate-quality-report--documentation)
- [3. Feature Engineering \& Feature Store](#3-feature-engineering--feature-store)
  - [3.1 Overview of Data Transformations \& Feature Engineering](#31-overview-of-data-transformations--feature-engineering)
  - [3.2 Key Data Transformations \& Feature Creation](#32-key-data-transformations--feature-creation)
    - [3.2.1 Creating Unique Identifiers](#321-creating-unique-identifiers)
    - [3.2.2 Numerical to Categorical Transformations](#322-numerical-to-categorical-transformations)
    - [3.2.3 Grouping Cities with low counts into "Other"](#323-grouping-cities-with-low-counts-into-other)
  - [In addition to cleaning the city names a new variable was created "Major City". The variable categorizing cities based on the top 5 cities by unique patient count, the remaining cities were labeled "Other" including rows with an "Unknown" vit value.](#in-addition-to-cleaning-the-city-names-a-new-variable-was-created-major-city-the-variable-categorizing-cities-based-on-the-top-5-cities-by-unique-patient-count-the-remaining-cities-were-labeled-other-including-rows-with-an-unknown-vit-value)
  - [3.3 Extracting Feature Tables](#33-extracting-feature-tables)
  - [3.4 Storing Features for Reuse](#34-storing-features-for-reuse)
- [4 Reports: City Level Analysis Report](#4-reports-city-level-analysis-report)
  - [4.1 Introduction](#41-introduction)
  - [4.2 Methodology](#42-methodology)
  - [4.3 Data Extraction](#43-data-extraction)
  - [4.4 Exploratory Data Analysis](#44-exploratory-data-analysis)
  - [4.5 Modeling](#45-modeling)
    - [4.5.1 : Choosing Model](#451--choosing-model)
    - [4.5.2 : Comparing Models](#452--comparing-models)
    - [4.5.3 : Model Results](#453--model-results)
    - [4.5.4 : Model Diagnostics](#454--model-diagnostics)
  - [](#)
  - [4.6 : Results](#46--results)
  - [](#-1)
  - [4.7 : Discussion and Limitations](#47--discussion-and-limitations)
  - [4.8 : Next Steps](#48--next-steps)
  - [4.9 : Conclusion](#49--conclusion)

<div style="page-break-after: always;"></div>

---

# 0. Executive Summary
This report presents an in-depth analysis of a diabetic population claims dataset to assess data quality, identify key trends, and apply advanced analytical techniques. The report outlines preprocessing steps, descriptive analytics, feature engineering, machine learning applications, and strategic recommendations for improving data usability and deriving actionable insights. The findings will support better decision-making in healthcare claims management.

<div style="page-break-after: always;"></div>

---
# 1. Technical Report Overview

## 1.1 Project Aim
This project aims to analyze claims data for a diabetic population, uncovering key trends, data quality issues, and predictive insights. The analysis demonstrates proficiency in data processing, feature engineering, and advanced analytics, ultimately supporting improved claims processing and risk assessment.


## 1.2 Objectives
- Showcase analytical and technological skills through systematic data exploration and predictive modeling.
- Improve data quality by identifying inconsistencies and recommending corrective measures.
- Identify patterns and trends in the diabetic population's claims data.
- Develop machine learning models to generate predictive insights.
- Propose strategies to enhance data integrity and analytics adoption within enterprise settings.

## 1.3 Skills Matrix
| Application in Report   | Description                                                                |
|-------------------------|----------------------------------------------------------------------------|
| Data Cleaning           | Handling missing values, duplicates, and plausibility checks               |
| Descriptive Analytics   | Summarizing dataset attributes, distribution, and trends                   |
| Data Quality Assessment | Identifying inconsistencies and proposing enhancements                     |
| Feature Engineering     | Creating structured and meaningful features for modeling                   |
| Advanced Analytics      | Applying clustering, predictive modeling, and hypothesis testing           |
| AI & LLM Integration    | Exploring retrieval-based AI insights for structured and unstructured data |
| Tools                   | Python, Pandas, Scikit-learn, Matplotlib, Seaborn, SQL, Parquet            |

*Suggestions: Add specific tools in the skills matrix*

## 1.4 Project Overview
This report demonstrates how a `modular analytics pipeline` improves data science workflows in an enterprise setting. The project focuses on analyzing a diabetes claims dataset, with tasks like data extraction, cleaning, feature engineering, model development, and deployment being handled independently by specialized teams.

In contrast to traditional siloed approaches, the modular pipeline enables parallelization, reducing bottlenecks and fostering efficient collaboration. The central component of this pipeline is a `feature store`, where engineered features are stored and made accessible for quick model development and deployment.
The pipeline includes the following stages:

- Data Extraction: Automated extraction from various sources like databases, APIs, and live events.
- Data Preprocessing: Data cleaning, validation, and initial EDA.
- Feature Engineering: Creation of features stored in the `feature store` for future use.
- Modeling and Reporting: Teams extract data from the `feature store`, build models, and perform hypothesis testing.
- Model Deployment & Monitoring: Final model deployment for generating predictive insights and supporting decision-making.

***Insert a graphics flow chart for the methodology section***

<div style="page-break-after: always;"></div>



# 2. Preprocessing Steps

## 2.1 Data Ingestion
<!-- ![Age Distribution](age_distribution.png) -->
The raw layer ingests CSV files while enforcing schema validation. This ensures that each column maintains the correct data type, preventing inconsistencies in downstream processing. The data is then converted to Parquet format for storage efficiency and type preservation.

**_Schema Validation and Column Standardization_**

To ensure data integrity, the ingestion process applies:
- Explicit data type enforcement using a predefined dictionary.
- Column renaming for consistency across analysis stages.	
- Initial data validation to verify data types and structure.

```python
# Define the data types for each column in the dataset
dtype_dict = {
    "MEMBER_CODE": "int64",    # De-identified member ID, stored as float to match dataset format
    "Age": "int64",             # Age of the member
    "GENDER": "category",       # Gender is a categorical variable
    "POLICY_NO": "int64",       # Policy number, stored as integer
    "CMS_Score": "int64",       # Charlson comorbidity index score, stored as integer
    "ICD_CODE": "category",     # ICD-10 codes are categorical
    "ICD_desc": "string",       # ICD-10 description as a string
    "City": "string",           # City as a string, handling missing values separately
    "CLAIM_TYPE": "category",   # Claim type is categorical
    "BMI": "float64"            # BMI as a float
}
```
<div style="page-break-after: always;"></div>

```python
# Column renaming lookup table
column_lookup = {
    "MEMBER_CODE": "member_code",
    "Age": "age",
    "GENDER": "gender",
    "POLICY_NO": "policy_number",
    "CMS_Score": "cms_score",
    "ICD_CODE": "icd_code",
    "ICD_desc": "icd_description",
    "City": "city",
    "CLAIM_TYPE": "claim_type",
    "BMI": "bmi"
}
# Load the dataset with specified data types
raw_data = pd.read_csv(raw_data_path, dtype=dtype_dict)

```

**_Conversion to Parquet for Type Preservation_**

After schema validation, the dataset is saved in `Parquet format` to maintain column types across different processing layers.

---
## 2.2 Data Cleaning
- The average age of patients is {df['Age'].mean():.2f} years.
- The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.

To ensure data consistency and integrity, a series of data cleaning steps were performed, including handling missing values, identifying and removing duplicates, and validating the dataset.

### 2.2.1 Identifying Missing Data

A summary of missing values was generated to assess completeness. The city column was identified as having 4,700 missing values, while other critical fields remained largely intact.

### 2.2.2 Handling Missing Data

The missing values in the city column were replaced with "Unknown" to retain records without introducing bias. Other imputation strategies, such as mean or mode imputation, were considered but were not required at this stage.


### 2.2.3 Checking for Duplicates

Duplicate records were reviewed to prevent redundant information.

- Duplicate rows found: 2, which were removed.
- The claim_type column exhibited duplication where records with an 'I' value had an identical counterpart with an 'O' value.
  - 34,233 out of 34,235 records with claim_type = I had an identical entry with claim_type = O.
  - These inconsistencies were flagged for further review and correction in subsequent processing stages.



### 2.2.4 Data Validation

Following the cleaning steps, a final validation was conducted to confirm:

- No remaining missing values in critical columns.
- No duplicate rows affecting dataset integrity.
- Flagging of claim_type anomalies for further processing.

This ensures the dataset is now structured, consistent, and ready for transformation in the next stage.

---
## 2.4 Data Quality checks
To ensure the dataset maintains logical consistency and accuracy, a series of data validation checks were conducted. These checks included logical value constraints, member data consistency assessments, and ICD code standardization. Any identified issues were documented for further analysis.

### 2.4.1 Logical Value Checks

Key numerical and categorical variables were validated against predefined logical ranges:
- age: Ensured all values fell within a plausible range (0–120 years).
- bmi: Confirmed all BMI values were within a reasonable range (10–80).
- gender: Ensured only expected categorical values ("M" or "F") were present.

*Suggestions: How will you handle outliers or incorrect values*

### 2.4.2 Identifying Family Units Using member_code

Some member_code values are linked to individuals of different ages and genders, suggesting that member_code may represent family units rather than individual identifiers. For example, the same member_code appears for both male and female individuals of varying ages, indicating family-level grouping within policies. A unique identifier for individuals is created in the feature engineering section.

### 2.4.3 Correct code mapping

An ICD-10 lookup table was created for icd_code and icd_description to:

- Identify duplicate or conflicting mappings.
- Ensure consistency in ICD-10 codes across records.
- Detect missing or mismatched descriptions requiring correction.

### 2.3.1 Issues with Claim Type into an Indicator

The claim_type column contained duplicate records where 'I' (initial claim) and 'O' (other claim) values represented the same entry. 
A transformation was applied to the claim_type variable to convert ‘I’ values into an indicator while handling duplicate rows:
- ‘I’ (Initial Claim) was transformed into a binary indicator variable.
- Duplicate entries caused by claim_type variations were identified and removed to ensure data integrity.


### 2.3.2. Non-Standardized City Names

The City names were not standardazied in a specific format. To ensure consistency in geographic analysis, city names were normalized by:
- Trimming whitespace to remove unwanted spaces.
- Converting names to title case (e.g., "RIYADH" → "Riyadh").
- Correcting hyphenation inconsistencies (e.g., "Al Khobar" → "Al-Khobar").
- Creating a lookup table to store standardized city names for reusability.

 
### 2.4.4 Intermediate Quality Report & Documentation

All flagged quality issues were documented and saved for further analysis. This included:

- Entries with out-of-range values (e.g., unrealistic BMI or age values).
- Household-based member_code groupings for further validation.
- ICD code standardization inconsistencies.

A summary of member_code groupings was saved in non_unique_member_codes.csv for further verification and future analytical segmentation.

<div style="page-break-after: always;"></div>

# 3. Feature Engineering & Feature Store

## 3.1 Overview of Data Transformations & Feature Engineering
This section outlines the structured process of transforming raw data into analytical features, ensuring consistency, standardization, and usability for downstream modeling and analysis. The approach follows a layered transformation method, converting raw attributes into structured feature tables.
Note: Feature engineering is an iterative process that evolves based on exploratory analysis. While this report presents feature creation before descriptive analytics for clarity, the actual process involved analyzing data gaps, transforming variables, and iterating based on insights.

---
## 3.2 Key Data Transformations & Feature Creation

### 3.2.1 Creating Unique Identifiers

To ensure accurate individual tracking while preserving privacy, a unique identifier was created by combining:
- policy_number
- member_code
- age
- gender

This identifier allows differentiation of individuals within the same policy while enabling household-level analysis.

### 3.2.2 Numerical to Categorical Transformations

To enhance interpretability and improve modeling performance, numerical variables were categorized into meaningful groups:

| Feature                         | Categories                                                                                      |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| **Age Group (age_group)**       | 10-year intervals (e.g., 0-9, 10-19, 20-29, …, 80+)                                             |
| **BMI Category (bmi_cat)**      | - Underweight (<18.5) <br> - Healthy (18.5–24.9) <br> - Overweight (25–29.9) <br> - Obese (≥30) |
| **Obesity Class (obesity_cat)** | - Class 1 (30–34.9) <br> - Class 2 (35–39.9) <br> - Class 3 (≥40)                               |

These categorizations allow for comparative risk analysis across different patient groups.


### 3.2.3 Grouping Cities with low counts into "Other" 

In addition to cleaning the city names a new variable was created "Major City". The variable categorizing cities based on the top 5 cities by unique patient count, the remaining cities were labeled "Other" including rows with an "Unknown" vit value.
---

## 3.3 Extracting Feature Tables

To enhance data accessibility, key feature tables were created for streamlined analysis:

| Feature Table               | Description                                                                     |
|-----------------------------|---------------------------------------------------------------------------------|
| **Diabetes Type Table**     | Aggregates diabetes classification based on ICD-10 codes per unique identifier. |
| **Comorbidity Table**       | Stores Charlson Comorbidity Index scores per unique identifier.                 |
| **Diabetes Feature Table**  | Captures diabetes-specific indicators, including treatment intensity.           |
| **Family Size Table**       | Groups members under the same policy to determine household size.               |
| **Unique Identifier Table** | Stores transformed IDs for downstream merging and validation.                   |
| **ICD-1 Lookup Table**      | A lookup table to stode the ICD-10 descriptions.                                |

---
## 3.4 Storing Features for Reuse
- Engineered features were saved in structured feature tables to facilitate efficient retrieval, reuse, and analysis.
- Outputs were stored in a feature store using Parquet format for optimized storage and column type consistency.

*Suggestions: Explain how feature tables will be used in subsequent analysis*

Feature tables will be used to streamline the modeling process by providing pre-processed, consistent, and reusable data. This ensures that all models and analyses are based on the same set of features, improving reproducibility and efficiency. For example, the diabetes feature table can be directly used to train machine learning models to predict complications, while the comorbidity table can be used to assess the impact of comorbid conditions on health outcomes.

note: not all the features may be used in future analysis. 

<div style="page-break-after: always;"></div>


# 4 Reports: City Level Analysis Report 

## 4.1 Introduction
The built environment—the design and layout of urban spaces—has long been associated with public health outcomes. While factors like access to healthcare services, availability of recreational spaces, and urban design are important, the geographic location of individuals within a city can also influence their health. In particular, for individuals living with chronic diseases like diabetes, various factors such as access to healthcare, lifestyle, and local healthcare resources can contribute to the progression of complications.

In this report, we investigate the association between the number of diabetes complications and the city-level location of diabetic patients within Saudi Arabia, using a sample of medical claims records. By examining the geographic distribution of diabetes complications and integrating demographic information (age, sex), body measurements (BMI), and comorbidity data, we aim to explore whether living in certain cities correlates with different health outcomes for diabetic individuals.

*The objectives of this analysis are:*

- Investigate the relationship between city-level location and the frequency of diabetes complications.
- Analyze demographic and health-related factors, including age, sex, BMI, and comorbidities, to understand their role in diabetes complications at the city level.
- Provide insights that may inform healthcare interventions or policy changes aimed at improving diabetes management and reducing complications across cities.
Ultimately, this analysis seeks to identify any significant patterns that could inform public health strategies tailored to specific geographic regions.

*Hypothesis:*

There is a significant relationship between the `number of diabetes complications claims` and the `city` in which the patients reside. The analysis will explore whether certain cities experience higher rates of complications, potentially influenced by demographic factors like `age`, `sex`, `BMI`, and `comorbidities`.



## 4.2 Methodology

This section outlines the steps for conducting the analysis, model building, and validation.

1. Data Extraction: Data is extracted from the feature store, ensuring consistency and accessibility for analysis.
2. Exploratory Data Analysis (EDA): A comprehensive analysis is performed, including:
   - Summary Statistics: To provide an overview of key variables.
   - Univariate Analysis: To analyze the distribution of individual variables.
   - Bivariate Analysis: To examine relationships between pairs of variables.
   - Multivariate Analysis: To explore interactions between multiple variables.
   - Correlation Analysis: To identify significant correlations between variables.
3. Model Building: The appropriate modeling techniques are selected based on the analysis.
4. Model Selection: The best-fitting model is chosen based on evaluation metrics such as AIC and BIC.
5. Model Evaluation: The model’s performance is assessed using metrics such as log-likelihood, pseudo-R², and other relevant statistical tests, along with residual diagnostics to check for model assumptions and ensure the reliability of the model.

## 4.3 Data Extraction

Using the feature store, we extracted the dataset `city_comp_df`, which focuses on individual-level demographics and claims summaries. The dataset includes the target variable of interest, number of reported diabetes complications, the main explanatory variable, city, and other key variables such as age, BMI, number of comorbidities, type of diabetes reported, and total number of diabetes reported. Below is a description of the variables included in the dataset:

| #  | Column              | Description                                                         | Dtype    |
|----|---------------------|---------------------------------------------------------------------|----------|
| 0  | unique_id           | Unique identifier for an individual                                 | object   |
| 1  | policy_number       | Policy number                                                       | int64    |
| 2  | member_code         | Member code                                                         | int64    |
| 3  | age_cat             | Age in category                                                     | category |
| 4  | age                 | Age in years                                                        | int64    |
| 5  | gender              | Gender category                                                     | category |
| 6  | max_bmi             | Maximum BMI recorded                                                | float64  |
| 7  | max_bmi_cat         | Maximum BMI in category                                             | category |
| 8  | max_major_city      | Most city by claims for an individual using the Major city category | object   |
| 9  | E09                 | ICD code for Impaired glucose regulation                            | float64  |
| 10 | E10                 | ICD code for type 1 diabetes mellitus                               | float64  |
| 11 | E11                 | ICD code for type 2 diabetes mellitus                               | float64  |
| 12 | E12                 | ICD code for malnutrition-related diabetes mellitus                 | float64  |
| 13 | E13                 | ICD code for other specified diabetes mellitus                      | float64  |
| 14 | E14                 | ICD code for unspecified diabetes mellitus                          | float64  |
| 15 | total_complications | Total number of diabetes complications claims                       | int64    |
| 16 | total_comorbidities | Total number of comorbidities claims                                | int64    |
| 17 | has_icd_dm          | Indicator if the individual has any ICD diabetes                    | int64    |
| 18 | total_dm_icd        | Total number of diabetes ICD codes reported                         | int64    |

<div style="page-break-after: always;"></div>

## 4.4 Exploratory Data Analysis

  
*Summary Statistics of numerical variables*

The table below provides a summary of key variables in the dataset:

| Variable             | mean  | std   | min   | 25%   | 50%   | 75%   | max   |
|----------------------|-------|-------|-------|-------|-------|-------|-------|
| age                  | 64.70 | 7.15  | 14.00 | 61.00 | 64.00 | 69.00 | 104.00|
| max_bmi              | 30.17 | 6.99  | 13.02 | 25.68 | 29.14 | 33.06 | 104.06|
| total_dm_icd         | 1.18  | 0.66  | 0.00  | 1.00  | 1.00  | 1.00  | 5.00  |
| total_complications  | 1.28  | 1.42  | 0.00  | 0.00  | 1.00  | 2.00  | 18.00 |
| total_comorbidities  | 1.03  | 0.75  | 0.00  | 1.00  | 1.00  | 1.00  | 6.00  |

In terms of the Total Complications:

On average, individuals have 1.28 reported diabetes complications of which 25% of individuals have zero reported complications and the maximum number of reported complications is 18, indicating a wide range of reported complications count among patients.  

*Total complications*
In addition around 35% of the population have no reported complication while 31% have 1 repoted complication [not shown in the previous table].

The distribution [image] shows a right-skewed pattern with a significant number of individuals reporting zero complications, followed by smaller frequencies for higher counts of complications. This suggests that most individuals experience low complication rates. The zero-inflated distribution indicates that many individuals report no complications, which should be considered when modeling (Poisson regression assumptions may be affected by the excess of zeros).

![Complications Distribution](image-15.png)


*City counts*

Jeddah has the highest count of individuals, representing 35.67% of the total population, followed by Riyadh. The 'Other' city goroup represents 23.52% of the population. This distribution shows a disproportionate representation of cities, with Jeddah and Riyadh having a significant portion of the data.

| Category | Count | Percentage |
|----------|-------|------------|
| Riyadh   | 5273  | 28.21      |
| Jeddah   | 6668  | 35.67      |
| Alkhobar | 885   | 4.73       |
| Madina   | 771   | 4.12       |
| Makkah   | 700   | 3.74       |
| Other    | 4397  | 23.52      |


![City counts](image-13.png)


*Age Distribution*

The age distribution is right-skewed, with noticeable increases in counts after the ages of 50 and 60. The largest proportion of the population falls within the 60-69 age group, representing 66.60% of the population, followed by the 70-79 age group at 20.31%. The average age is 64.7 years. The 50-59 age group accounts for 9.79%, while other age groups each represent less than 2% of the population.


| Category | Count | Percentage |
|----------|-------|------------|
| 10-19    | 3     | 0.02       |
| 20-29    | 18    | 0.10       |
| 30-39    | 121   | 0.65       |
| 40-49    | 230   | 1.23       |
| 50-59    | 1831  | 9.79       |
| 60-69    | 12450 | 66.60      |
| 70-79    | 3797  | 20.31      |
| 80-89    | 218   | 1.17       |
| 90-99    | 25    | 0.13       |
| 100-109  | 1     | 0.01       |

![Age distribution](image-5.png)

![Age category disribution](image-4.png)


***BMI Distribution***

The BMI distribution is right-skewed, with a majority of individuals (> 77%) falling within the overweight and obese categories.

| Category    | Count | Percentage |
|-------------|-------|------------|
| Underweight | 154   | 0.82       |
| Healthy     | 4007  | 21.43      |
| Overweight  | 6440  | 34.45      |
| Obesity     | 8093  | 43.29      |


![BMI distribution](image-6.png)
![BMI category barsplot](image-7.png)



*comorbidities*

The majority of individuals have one comorbidity, followed by a smaller portion with zero comorbidities and few with more than on.

![alt text](image-16.png)

*Total reported diabetes type per patient*

The majority of individuals have one reported diabets diagnosis, while a large proportion of individuals have no reported diabetes type (plausabile explanation is that this is a sampled from a larger claims data set and individuals with a diabetes claim might not be within the smaples extracted).

![alt text](image-14.png)

<div style="page-break-after: always;"></div>

*Correlations Analysis*

Total complications is positively correlated with E10, E11, and E13 (types of diabetes), indicating that individuals with these diagnoses tend to have more complications.
There is a strong correlation between total complications and total DM ICD codes. City vairables show a weak correlation with total diabetes complications, with Jeddah being the only cities showing a moderate positive correlation. Age showes a weak negative correlation while BMI, and comorbidities also show weak positive association with complications. 

![alt text](image-24.png)


*Total Complications by City*

The boxplot shows the distribution of total diabetes complications across cities. There is no clear difference in the total complications across cities.

![boxplot city](image-30.png)

There is no visible difference in the distribution of total complications between cities when stratified by gender. However, female patients exhibit higher variability in complication levels compared to males, except in AlKhobar and Madina, where the variability between genders appears similar. this may suggest that gender may influence the degree of complications, with women having more variance in the total complications.

![boxplot city by gender](image-31.png)

When stratifying across bmi groups we observe that higher BMI (especially obesity) is linked to more total complications compared to healthyacross cities, and underweight individuals also show a slight tendency for higher complications but with less variability, though this relationship is weak.

![boxplot city by bmi cat](image-35.png)

More comorbidities are associated with higher total diabets complications across cities.

![boxplot city by comorbidities](image-33.png)

There is a very strongassociation between increasing number of different diabetes diagnosis on an individual and the total diabetes comorbidities. This association is consistant across all cities. 

![boxplot city by total reported diabetes types](image-34.png)

<div style="page-break-after: always;"></div>


## 4.5 Modeling



### 4.5.1 : Choosing Model
This section is critical because it explains why you chose a particular modeling technique over others. You might want to expand on:
- Rationale: Why Poisson regression? Was it due to the nature of the data (count data for diabetes complications)? Were there other methods considered (e.g., Negative Binomial, Logistic Regression for binary outcomes)?
- Assumptions: Discuss any assumptions the model makes. For Poisson regression, you can talk about the assumption of equal variance and mean. If those assumptions don't hold, you might need to discuss the need for a different model (e.g., Negative Binomial for overdispersed data).
- Model Selection Criteria: What was the key factor for selecting the model (predictive power, interpretability, handling of specific types of data, etc.)?

### 4.5.2 : Comparing Models
This section should provide a comparison of multiple models, and the following points might be helpful:
- Baseline Model: Start with a simple baseline model (e.g., univariate regression with just one predictor). This will help you assess the improvement when adding more features.
- Additional Models: Compare the chosen model (Poisson regression) with alternatives like:
A simple linear regression (if the outcome were continuous),
Generalized Linear Models (GLM) with a different distribution (e.g., Negative Binomial),
Decision trees, Random Forests, or XGBoost if you want to show a machine learning approach.
- Model Metrics: Evaluate each model based on common performance metrics such as:
  - AIC/BIC (for model comparison),
  - Log-likelihood,
  - Deviance (for Poisson models),
  - RMSE, R-squared, or other appropriate metrics depending on the model type.
  - Cross-validation scores to ensure robustness of the model.


| Coefficient            | Intercept         | Model 1           | Model 2           | Model 3           | Model 4           |
|------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| inflate_const          | -1.550*** (0.031) | -1.686*** (0.034) | -1.819*** (0.037) | -1.898*** (0.040) | -14.902 (87.362)  |
| Intercept              | 0.440*** (0.007)  | 0.270*** (0.014)  | 0.996*** (0.072)  | 0.859*** (0.072)  | -0.346*** (0.068) |
| City = Alkhobar        |                   | 0.205*** (0.034)  | 0.194*** (0.034)  | 0.183*** (0.034)  | 0.237*** (0.032)  |
| City = Jeddah          |                   | 0.308*** (0.018)  | 0.302*** (0.018)  | 0.260*** (0.018)  | 0.132*** (0.017)  |
| City = Madina          |                   | 0.135*** (0.039)  | 0.126*** (0.038)  | 0.067* (0.038)    | -0.002 (0.035)    |
| City = Makkah          |                   | 0.273*** (0.038)  | 0.220*** (0.038)  | 0.167*** (0.038)  | 0.194*** (0.035)  |
| City = Other           |                   | -0.005 (0.021)    | -0.007 (0.021)    | -0.019 (0.021)    | -0.017 (0.020)    |
| Gender = M             |                   |                   | 0.054*** (0.014)  | 0.011 (0.015)     | 0.058*** (0.014)  |
| Age                    |                   |                   | -0.017*** (0.001) | -0.017*** (0.001) | -0.010*** (0.001) |
| BMI                    |                   |                   | 0.011*** (0.001)  | 0.009*** (0.001)  | 0.006*** (0.001)  |
| Total Comorbidities    |                   |                   |                   | 0.205*** (0.009)  | 0.135*** (0.008)  |
| Total DM Types claimed |                   |                   |                   |                   | 0.617*** (0.008)  |
| N              | 18694      | 18694      | 18694      | 18694      | 18694      |
| AIC            | 58166.836  | 57748.263  | 57324.429  | 56792.155  | 51987.466  |
| BIC            | 58182.508  | 57803.114  | 57402.789  | 56878.351  | 52081.497  |
| Log-Likelihood | -29081.418 | -28867.131 | -28652.215 | -28385.078 | -25981.733 |
| Pseudo R2      | 0.000      | 0.007      | 0.015      | 0.024      | 0.107      |

*Notes:*
Standard errors in parentheses [** p<.05, ***p<.01]

### 4.5.3 : Model Results

Summarize the key results of the modeling in this section:
- We present 4 models with increasing complexity as presented in the following table
- The model with highest explainability (model 4)
- The variability explained (low)
- The findings
  - Cities coefficient comparison
  - Other important variables
- Overal model performance
- Model performance: How well did the model perform in terms of goodness of fit and predictive power? Show key statistics such as AIC, BIC, and deviance residuals.
- Model diagnostics: Were there any issues detected (e.g., residuals, goodness of fit, cross-validation results, outliers, multicollinearity)? (next sections)

### 4.5.4 : Model Diagnostics

Residual diagnostics of model 4

![alt text](image-21.png)
![alt text](image-22.png)
![alt text](image-23.png)

*conclusion*
- Significant deviation at the tails
- residuals not randomly distributed
- The model does not show overall fit
- Recommendation: 
  - Transformation (difficult since this is count variabel)
  - Other modeling techniques ()
  - Add more data and variables for explainability

<div style="page-break-after: always;"></div>
---

## 4.6 : Results
Interpretability and Insights:
- Explainability is critical in healthcare data:
  - Differences in probability of comorbidities across geographical locations may have and underlying cause that need to be investigated further (accessibility to timely care, adherence to best practices, patient education ...etc.)
  - Targeted complication prevention programs 
  - 
- If you're using Poisson regression, explain the coefficients (e.g., a one-unit increase in BMI leads to a certain increase in the expected count of complications).
- Impact of individual predictors: Provide real-world interpretation of how different factors affect the outcome (e.g., how does age or comorbidity score influence complications?).
  
- Differences in probability of comorbidities across geographical locations may have and underlying cause that need to be investigated further (accessibility to timely care, adherence to best practices, patient education ...etc.)  
- If cities with certain demographics are identified as higher risk, targeted interventions could be proposed.

Strategic Recommendations: Based on the insights, propose strategies for improving healthcare claims management or preventive measures for diabetes-related complications. Recommendations could include:
- Tailored interventions in high-risk areas.
- Improving data quality or feature collection (e.g., including additional demographic information).

**Emphasizing the Business Value: While you've done a good job focusing on technical aspects, be sure to tie back findings and recommendations to the business or healthcare outcomes.**

<div style="page-break-after: always;"></div>
---

## 4.7 : Discussion and Limitations
This section is crucial for reflecting on the analysis. You should address:
- Model Assumptions: Were any assumptions (e.g., for Poisson regression) violated, and how did you handle them? For example, if the Poisson model assumption of mean variance doesn’t hold, how did you adjust (e.g., switching to a Negative Binomial model)?
  - Model assumption did not hold (zero inflations and overdispertion)
  - for zero inflation we adjusted the mode for a zeroinflationPoisson regression
  - For overdisperstion we opted to ignore since the zero binoal model did not converge
- Data Issues: Any potential data issues that could have influenced the model (e.g., missing data, measurement error)?
  - Limited data
  - measurment errors are probable (repeated claims not clear)
  - over representation from specific cities
  - grouping of low count cities
- External Factors: Are there any external factors that could affect the interpretation of the results, such as socioeconomic factors, healthcare system differences, or unmeasured variables?
  - The model did not include external factors that migh help in explainability including but not limited to [policy type, socioeconomic factors, ethnicity, living location and other geographical information]
  - 
- Model Limitations: Are there any limitations of the model that could affect its generalizability or applicability in real-world decision-making?
  - The model had a poor fit and low variance explainability score
  - 
---

## 4.8 : Next Steps
Further Analysis: What additional analyses could be conducted to build on this model? You could consider:
- Exploring interactions between features (e.g., age × BMI).
- Using more sophisticated models (e.g., Random Forest or Gradient Boosting) to predict diabetes complications with greater accuracy.
- Incorporating time-series data (if available) for longitudinal predictions.
- Data Collection and Enhancements: Propose further data collection that could enhance the modeling process, such as more granular data on lifestyle factors, treatments, etc.
- Model Deployment: If the model is deemed reliable and useful, discuss potential deployment into real-world applications (e.g., integrating with healthcare systems for predicting claims).
---
  
## 4.9 : Conclusion
The conclusion should summarize the key takeaways from the analysis:
- Restate the modeling approach and summarize which model worked best.
- Mention the key predictors that were most influential (age, BMI, etc.).
- Highlight the implications for healthcare decision-making or claims management.
