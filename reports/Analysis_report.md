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
  - [2.4 Data Quality checks](#24-data-quality-checks)
- [3. Feature Engineering \& Feature Store](#3-feature-engineering--feature-store)
  - [3.1 Overview of Data Transformations \& Feature Engineering](#31-overview-of-data-transformations--feature-engineering)
  - [3.2 Key Data Transformations \& Feature Creation](#32-key-data-transformations--feature-creation)
  - [3.3 Extracting Feature Tables](#33-extracting-feature-tables)
  - [3.4 Storing Features for Reuse](#34-storing-features-for-reuse)
- [4 Reports: City Level Analysis Report](#4-reports-city-level-analysis-report)
  - [4.1 Introduction](#41-introduction)
  - [4.2 Methodology](#42-methodology)
  - [4.3 Data Extraction](#43-data-extraction)
  - [4.4 Exploratory Data Analysis](#44-exploratory-data-analysis)
  - [4.5 Modeling](#45-modeling)
  - [](#)
  - [4.6 : Model Results](#46--model-results)
  - [4.7 Discussion, Limitations](#47-discussion-limitations)
  - [4.9 : Conclusion and Strategic Recommendations](#49--conclusion-and-strategic-recommendations)

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
| Data Ingestion          | Best practifce in data loading and data type.                              |
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

***2.2.1 Identifying Missing Data***

A summary of missing values was generated to assess completeness. The city column was identified as having 4,700 missing values, while other critical fields remained largely intact.

***2.2.2 Handling Missing Data***

The missing values in the city column were replaced with "Unknown" to retain records without introducing bias. Other imputation strategies, such as mean or mode imputation, were considered but were not required at this stage.


***2.2.3 Checking for Duplicates***

Duplicate records were reviewed to prevent redundant information.

- Duplicate rows found: 2, which were removed.
- The claim_type column exhibited duplication where records with an 'I' value had an identical counterpart with an 'O' value.
  - 34,233 out of 34,235 records with claim_type = I had an identical entry with claim_type = O.
  - These inconsistencies were flagged for further review and correction in subsequent processing stages.



***2.2.4 Data Validation***

Following the cleaning steps, a final validation was conducted to confirm:

- No remaining missing values in critical columns.
- No duplicate rows affecting dataset integrity.
- Flagging of claim_type anomalies for further processing.

This ensures the dataset is now structured, consistent, and ready for transformation in the next stage.


## 2.4 Data Quality checks
To ensure the dataset maintains logical consistency and accuracy, a series of data validation checks were conducted. These checks included logical value constraints, member data consistency assessments, and ICD code standardization. Any identified issues were documented for further analysis.

***2.4.1 Logical Value Checks***

Key numerical and categorical variables were validated against predefined logical ranges:
- age: Ensured all values fell within a plausible range (0–120 years).
- bmi: Confirmed all BMI values were within a reasonable range (10–80).
- gender: Ensured only expected categorical values ("M" or "F") were present.

*Suggestions: How will you handle outliers or incorrect values*

***2.4.2 Identifying Family Units Using member_code***

Some member_code values are linked to individuals of different ages and genders, suggesting that member_code may represent family units rather than individual identifiers. For example, the same member_code appears for both male and female individuals of varying ages, indicating family-level grouping within policies. A unique identifier for individuals is created in the feature engineering section.

***2.4.3 Correct code mapping***

An ICD-10 lookup table was created for icd_code and icd_description to:

- Identify duplicate or conflicting mappings.
- Ensure consistency in ICD-10 codes across records.
- Detect missing or mismatched descriptions requiring correction.

***2.4.4 Issues with Claim Type into an Indicator***

The claim_type column contained duplicate records where 'I' (initial claim) and 'O' (other claim) values represented the same entry. 
A transformation was applied to the claim_type variable to convert ‘I’ values into an indicator while handling duplicate rows:
- ‘I’ (Initial Claim) was transformed into a binary indicator variable.
- Duplicate entries caused by claim_type variations were identified and removed to ensure data integrity.


***2.3.2. Non-Standardized City Names***

The City names were not standardazied in a specific format. To ensure consistency in geographic analysis, city names were normalized by:
- Trimming whitespace to remove unwanted spaces.
- Converting names to title case (e.g., "RIYADH" → "Riyadh").
- Correcting hyphenation inconsistencies (e.g., "Al Khobar" → "Al-Khobar").
- Creating a lookup table to store standardized city names for reusability.

 
***2.4.4 Intermediate Quality Report & Documentation***

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

***3.2.1 Creating Unique Identifiers***

To ensure accurate individual tracking while preserving privacy, a unique identifier was created by combining:
- policy_number
- member_code
- age
- gender

This identifier allows differentiation of individuals within the same policy while enabling household-level analysis.

***3.2.2 Numerical to Categorical Transformations***

To enhance interpretability and improve modeling performance, numerical variables were categorized into meaningful groups:

| Feature                         | Categories                                                                                      |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| **Age Group (age_group)**       | 10-year intervals (e.g., 0-9, 10-19, 20-29, …, 80+)                                             |
| **BMI Category (bmi_cat)**      | - Underweight (<18.5) <br> - Healthy (18.5–24.9) <br> - Overweight (25–29.9) <br> - Obese (≥30) |
| **Obesity Class (obesity_cat)** | - Class 1 (30–34.9) <br> - Class 2 (35–39.9) <br> - Class 3 (≥40)                               |

These categorizations allow for comparative risk analysis across different patient groups.


***3.2.3 Grouping Cities with low counts into Other category***

In addition to cleaning the city names a new variable was created "Major City". The variable categorizing cities based on the top 5 cities by unique patient count, the remaining cities were labeled "Other" including rows with an "Unknown" vit value.


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

  
***4.4.1. Summary Statistics of numerical variables***

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

***4.4.2 Total complications***

In addition around 35% of the population have no reported complication while 31% have 1 repoted complication [not shown in the previous table].

The distribution [image] shows a right-skewed pattern with a significant number of individuals reporting zero complications, followed by smaller frequencies for higher counts of complications. This suggests that most individuals experience low complication rates. The zero-inflated distribution indicates that many individuals report no complications, which should be considered when modeling (Poisson regression assumptions may be affected by the excess of zeros).

![Complications Distribution](image-15.png)


***4.4.3 City counts***

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


***4.4.4 Age Distribution***

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


***4.4.5 BMI Distribution***

The BMI distribution is right-skewed, with a majority of individuals (> 77%) falling within the overweight and obese categories.

| Category    | Count | Percentage |
|-------------|-------|------------|
| Underweight | 154   | 0.82       |
| Healthy     | 4007  | 21.43      |
| Overweight  | 6440  | 34.45      |
| Obesity     | 8093  | 43.29      |


![BMI distribution](image-6.png)
![BMI category barsplot](image-7.png)



***4.4.5 Total Comorbidities***

The majority of individuals have one comorbidity, followed by a smaller portion with zero comorbidities and few with more than on.

![alt text](image-16.png)

***4.4.6 Total reported diabetes type per patient***

The majority of individuals have one reported diabets diagnosis, while a large proportion of individuals have no reported diabetes type (plausabile explanation is that this is a sampled from a larger claims data set and individuals with a diabetes claim might not be within the smaples extracted).

![alt text](image-14.png)

<div style="page-break-after: always;"></div>

***4.4.7 Correlations Analysis***

Total complications is positively correlated with E10, E11, and E13 (types of diabetes), indicating that individuals with these diagnoses tend to have more complications.
There is a strong correlation between total complications and total DM ICD codes. City vairables show a weak correlation with total diabetes complications, with Jeddah being the only cities showing a moderate positive correlation. Age showes a weak negative correlation while BMI, and comorbidities also show weak positive association with complications. 

![alt text](image-24.png)


***4.4.8 Total Complications by City***

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

***4.5.1 Model Slection***

The selection of the model was driven by several crucial factors:

- Nature of the Analysis: Our analysis is primarily focused on hypothesis testing, where explainability is prioritized over pure predictive performance. The goal is to understand the key factors driving diabetes complications, rather than solely optimizing predictive accuracy. This necessitated the choice of a model that would provide both insight and interpretability of the results.

- Count Nature of the Target Variable: The target variable, total diabetes complications, represents count data (i.e., the number of complications per individual). This type of outcome is naturally suited for count regression models. Given the characteristics of our target variable, we considered models such as Poisson regression, which is commonly used for modeling count data.

- Model Assumptions: Poisson regression assumes equidispersion, where the mean and variance of the outcome variable are equal. However, in practice, this assumption may not always hold, especially in datasets with significant skewness or excess zeros.

**Assumption Checks:**

To test whether the assumptions of Zero-Inflation and Equidispersion hold, we performed the following checks:

1. Zero-Inflation:
   - We examined summary statistics for the number of claims and calculated the percentage of zero claims.
   - A visual inspection of the histogram of the number of claims was also conducted to assess the presence of a large number of zeros, which is indicative of zero-inflation.
2. Dispersion Statistic:
   - We calculated the variance-to-mean ratio. A value greater than 1 suggests overdispersion (where the variance exceeds the mean), indicating that Poisson regression might not be the most suitable model. The dispersion statistic for the dataset was 1.56, which is greater than 1, indicating overdispersion (see below image). This overdispersion is likely driven by the large number of zero values (many individuals reporting no complications).

To address this, we compared two potential models: the Zero-Inflated Poisson (ZIP) model and the Zero-Inflated Negative Binomial (ZINB) model. Although the ZINB model would theoretically provide a better fit due to its ability to handle overdispersion, it failed to converge during model fitting. As a result, we proceeded with the Zero-Inflated Poisson (ZIP) model.

![alt text](image-36.png)


***4.5.2 Model building***

Following the model selection process, several models of increasing complexity were built to analyze the factors influencing total diabetes complications. The models are compared based on their goodness of fit, AIC, BIC, and Pseudo R² values.

The models built include:
- Model 1: A basic model with city-level effects.
- Model 2: Adds gender, age and BMI as additional predictors.
- Model 3: Adds total comorbidities as additional predictors.
- Model 4: Adds the total diabetes types claimed.

Model Comparison Criteria:
- Goodness of Fit: This is assessed by examining how well the model fits the data.
- AIC and BIC: Lower values indicate better model fit, with Model 4 performing best across both metrics.
- Pseudo R²: This metric increases with model performance, and Model 4 achieves the highest Pseudo R² value (0.107), indicating the best fit and explanatory power.


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

***4.5.3  Model Comparison & Diagnostics***

**Model Comparison:**

Model 4 explained the highest proportion of variability, with a Pseudo R² of 0.107. While this is the best among the models tested, it remains relatively low, indicating room for improvement. Enhancing the model could involve selecting a more appropriate approach or adding more relevant data to better explain the variability.

Several robust models, including those with interaction terms and clustered standard errors by policy_number, were tested, but no significant improvement in model quality was observed [results not shown here].

In terms of model selection criteria, Model 4 showed the best performance with the lowest AIC (51,987.466) and BIC (52,081.497), indicating a superior fit. However, despite these advantages, the model's overall predictive power remains modest, suggesting that further refinement is needed.

**Residual Diagnostics**

The Residuals vs Fitted Values plot for model 4 shows that residuals are not randomly distributed but exhibit no clear biases or outliers.

![alt text](image-21.png)

The Q-Q plot reveals deviations from normality, particularly in the right tail, indicating poor fit for extreme values. This is likely due to the count nature of the target variable (Poisson or Negative Binomial distribution with zero inflation).

![alt text](image-22.png)

centration around zero with slight right skew, suggesting the model underpredicts extreme values. The concentration near zero may also reflect the zero-inflated nature of the data.

![alt text](image-23.png)

Overall the model diagnostic indicate that the model has difficulty fitting extreme value. 

<div style="page-break-after: always;"></div>
---

## 4.6 : Model Results

***4.6.1 City-Level Effects (Main Predictor)***

Model 4 shows that Alkhobar, Jeddah, and Makkah have a positive association with the total number of diabetes complications compared to the baseline city, Riyadh. The coefficients for these cities are as follows:
- Alkhobar: Coefficient = 0.237 (p < 0.01)
- Jeddah: Coefficient = 0.132 (p < 0.01)
- Makkah: Coefficient = 0.194 (p < 0.01)

In contrast, Madina and the Other cities category do not show significant associations with diabetes complications after controlling for other factors such as age, gender, BMI, total comorbidities, and total diabetes types claimed.

***4.6.2 Other Key Predictors***

Key predictors with strong associations to the target variable include Total Diabetes Types Claimed, 
with a coefficient of 0.617 (p < 0.01), and Total Comorbidities, with a coefficient of 0.135 (p < 0.01).
The predictors Age, BMI, and Gender all showed a weak association significant association.

***4.6.2 Interpretability of the model***

Although a regulare Poisson model directly interprets how predictors affect the count of complications (e.g. each unit increase in BMI increases the expected count of complications by x amount) the zero inflated model is more complicated. In this case we focus on the direction of association between the predictor variables and the target in our interpretation of the model as opposed to the direct quantification of the association. 


<div style="page-break-after: always;"></div>


## 4.7 Discussion, Limitations

***4.7.1 Discussion***

In this analysis, we hypothesized that there is an association between the number of diabetes complications and city location, based on previous studies that have highlighted the impact of urban environments on health outcomes. The findings of this analysis do not contradict this hypothesis. Specifically, we observed differences in the expected counts of diabetes complications across cities, with Alkhobar, Jeddah, and Makkah showing higher expected counts compared to the baseline city, Riyadh.
However, while these differences are statistically significant, they cannot be fully attributed to city-level effects alone. Several limitations affect the interpretation of the results:
- **Missing Individual-Level Factors**: The model did not account for key individual-level factors that are known to influence diabetes complications, such as socioeconomic status, education level, physical activity, other behavioral risk factors, and type of policy. Additionally, other medical history factors, including the length of diabetes and quality of care received, may also play a significant role in the variation of diabetes complications across cities. These factors were not included in the analysis due to data limitations.
- **Unmeasured City-Level Exposure Factors**: Within each city, exposure factors such as population density distribution, access to healthcare, neighborhood walkability scores, and levels of pollution could significantly impact health outcomes. These environmental factors are largely unknown or unmeasured in this dataset, and their influence could contribute to the observed differences in diabetes complications across cities.
- **Unequal Representation of City Populations**: The data sample is not equally representative of different population strata across cities. The sampling method is not well-documented, and the population distribution, as seen in the EDA, suggests an overrepresentation of certain cities like Jeddah and Riyadh, while other cities may be underrepresented. This unequal sampling could bias the results and affect the generalizability of the findings.

In addition to the city-level effects, we also observed a strong association between the number of different diabetes types reported and the total reported complications. This finding may be explained by several factors, including:
- **Unclear Diabetes Diagnosis**: Individuals with an unclear or improperly categorized diabetes type may experience worse outcomes due to misdiagnosis or lack of tailored treatment.
- **Inconsistent Reporting**: The variability in reporting different diabetes types could be indicative of low-quality care or inconsistent documentation practices, leading to incomplete or inaccurate treatment records.
- **Unrecognized Special Diabetes Types**: There may be special types of diabetes that are underreported or misclassified, which could impact patient management and complicate outcomes.
- **Other Factors**: Additional factors, such as coexisting medical conditions, treatment regimens, or unmeasured aspects of healthcare delivery, could also play a role and warrant further investigation.
This strong association underscores the importance of accurate diabetes classification and comprehensive patient management in understanding and reducing complications. Further research is needed to explore these potential explanations and their implications for improving care and health outcomes.

***4.7.2 Limitations***

Based on the findings of the analysis, the following recommendations are suggested:
- **Sampling limitations**:The data has limited representation at both the individual and city levels, with an unclear and unequal sampling method. This is reflected in the population distribution, which shows overrepresentation of cities like Jeddah and Riyadh, and underrepresentation of others. The sampling method is not well-documented, potentially introducing bias into the results.
- **Individual-Level factors**: Important individual-level factors, such as socioeconomic status, education level, physical activity, and quality of care received, were not included in the model. These factors are known to be strongly associated with diabetes complications and their absence likely impacts the model's predictive power. The length of diabetes and medical history were also not considered, both of which may play significant roles in determining complication outcomes.
- **External factors**: Many external factors that could affect the probability of complications were not accounted for in the model. These include socioeconomic status, healthcare access, policy type, and neighborhood-level factors such as walkability and pollution. These missing variables likely contribute to the low explainability of the model and the unexplained variation in complications across cities. Further research is needed to incorporate these factors into future models for better interpretability.
- **Modeling limitations**: The Zero-Inflated Poisson (ZIP) model was used to account for zero-inflation and overdispersion. However, these issues led to a poor model fit, especially with extreme values being underfitted, as indicated by the residual analysis. The low variance explanation of the model suggests that the predictors used were insufficient in explaining the variation in diabetes complications.

  
## 4.9 : Conclusion and Strategic Recommendations

This analysis examined the relationship between city-level location and diabetes complications, highlighting significant differences in complications across cities. The model also identified Total Diabetes Types Claimed and Total Comorbidities as strong predictors of complications, underscoring the importance of accurate diagnoses and comprehensive patient management.
While the findings align with existing literature on the built environment’s impact on health outcomes, the analysis was limited by several factors, including data quality, sampling bias, and unmeasured external factors. Future analyses should address these limitations by incorporating more granular data, exploring additional predictors, and using more sophisticated models for improved accuracy.
Strategic recommendations focus on targeted interventions in high-risk areas, enhancing data collection, and investigating the built environment and healthcare quality to better understand the drivers of diabetes complications. Implementing these recommendations will help improve healthcare delivery and preventive care for diabetes patients.

***A summary of the Strategic Recommendations:***

1. Targeted Interventions in High-Risk Areas
Implement tailored interventions in high-risk geographic locations (e.g., within Jeddah, Alkhobar and Makkah) based on demographics and complication rates. Focus on preventive care, health education, and higher quality diabetes management.
2. Enhance Data Quality and Feature Collection
Improve data collection by including demographic, geographical, and health behavior information. Investigate inconsistencies, particularly in cases where multiple diabetes types are reported for a single patient.
3. Investigate Built Environment Impact
Explore how environmental factors (e.g., walkability, proximity to healthcare, air pollution) impact diabetes complications through GIS data and spatial analysis.
4. Improve Healthcare Quality and Access
Assess the quality of care and identify any regional variations in diabetes complication rates across healthcare providers and locations.
  