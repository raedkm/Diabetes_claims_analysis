<div style="display: flex; flex-direction: column; justify-content: center; height: 100vh; text-align: center;">
  <h1>Analysis of Diabetic Population Claims Data</h1>
  <h3>Author: Raed K. Alotaibi</h3>
</div>

<div style="page-break-after: always;"></div>

## Table of Contents
- [0. Executive Summary](#0-executive-summary)
- [1. Technical Report Overview](#1-technical-report-overview)
  - [1.1 Project Aim](#11-project-aim)
  - [1.2 Objectives](#12-objectives)
  - [1.3 Skills Matrix](#13-skills-matrix)
  - [1.4 Project Overview](#14-project-overview)
- [2. Preprocessing Steps](#2-preprocessing-steps)
  - [2.1 Project setups and tools](#21-project-setups-and-tools)
  - [2.2 Data Ingestion](#22-data-ingestion)
  - [2.3 Data Cleaning](#23-data-cleaning)
  - [2.4 Data Quality checks](#24-data-quality-checks)
- [3. Feature Engineering \& Feature Store](#3-feature-engineering--feature-store)
  - [3.1 Overview](#31-overview)
  - [3.2 Key Data Transformations \& Feature Creation](#32-key-data-transformations--feature-creation)
  - [3.3 Extracting Feature Tables](#33-extracting-feature-tables)
  - [3.4 Storing Features for Reuse](#34-storing-features-for-reuse)
- [4 Reports: City Level Analysis Report](#4-reports-city-level-analysis-report)
  - [4.1 Introduction](#41-introduction)
  - [4.2 Methodology](#42-methodology)
  - [4.3 Data Extraction](#43-data-extraction)
  - [4.4 Exploratory Data Analysis](#44-exploratory-data-analysis)
  - [4.5 Modeling](#45-modeling)
  - [4.6 Model Results](#46-model-results)
  - [4.7 Discussion](#47-discussion)
  - [4.9 Conclusion and Strategic Recommendations](#49-conclusion-and-strategic-recommendations)

<div style="page-break-after: always;"></div>

---

# 0. Executive Summary
This report presents an analysis of a diabetes claims dataset to showcase the candidate’s expertise in analytics, 
reasoning, and technology. The project follows best practices in data engineering, feature engineering, and 
model development to generate actionable healthcare insights.

The approach employs a structured data engineering process that ensures data integrity and quality, with a feature store as a 
key element providing consistency and reusability across models, improving efficiency and reproducibility.

The primary hypothesis proposed that city-level location influences the occurrence of diabetes complications. This hypothesis is supported by strong scientific evidence suggesting that the built environment—factors like urban design, access to healthcare, and lifestyle—can have a substantial impact on long-term health outcomes. 

The hypothesis was tested using a Zero-Inflated Poisson (ZIP) regression model. The model aimed to predict the number of reported diabetes complication claims per patient, with city location as the main explanatory variable, while controlling for important variables such as age, gender, BMI, and comorbidities. The model was selected for its ability to handle zero-inflation and its focus on explainability

The results revealed significant differences in complication rates across cities, with Jeddah and Alkhobar exhibiting higher rates than Riyadh. Additionally, a strong association was found between the total number of different diabetes types reported for a patient and the total number of diabetes complications.

Based on these findings, several strategic recommendations are proposed. Health interventions should be targeted in high-risk areas within cities , where higher complication rates were observed, with a focus on improving diabetes care and reducing complications in these areas. Additionally, enhancing the quality of care in high risk areas is crucial, addressing gaps in healthcare services, patient management, and access to medical resources. Furthermore, improving data collection practices is essential, particularly by incorporating geographical information, socioeconomic status, and lifestyle factors. This will enable more detailed and accurate analyses, helping to refine predictive models and provide personalized insights to guide healthcare planning and resource allocation.

<div style="page-break-after: always;"></div>


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
| Application in Report      | Description                                                                |
|----------------------------|----------------------------------------------------------------------------|
| Data Ingestion             | Best practifce in data loading and data type.                              |
| Data Cleaning              | Handling missing values, duplicates, and plausibility checks               |
| Descriptive Analytics      | Summarizing dataset attributes, distribution, and trends                   |
| Data Quality Assessment    | Identifying inconsistencies and proposing enhancements                     |
| Feature Engineering        | Creating structured and meaningful features for modeling                   |
| Advanced Analytics         | Applying clustering, predictive modeling, and hypothesis testing           |
| Tools                      | Python, Pandas, Scikit-learn, Matplotlib, Seaborn, SQL, Parquet            |
| AI & LLM Integration [WIP] | Exploring retrieval-based AI insights for structured and unstructured data |

## 1.4 Project Overview
This report demonstrates how a `modular analytics pipeline` improves data science workflows in an enterprise setting. The project focuses on analyzing a diabetes claims dataset, with tasks like data extraction, cleaning, feature engineering, model development, and deployment being handled independently in different layers.

In contrast to traditional siloed approaches, the modular pipeline enables parallelization, reducing bottlenecks and fostering efficient collaboration. The central component of this pipeline is a `feature store`, where engineered features are stored and made accessible for quick model development and deployment.
The pipeline includes the following stages:

- Data Extraction: Automated extraction from various sources like databases, APIs, and live events.
- Data Preprocessing: Data ingestion, data cleaning, validation, and initial EDA.
- Feature Engineering: Creation of features stored in the `feature store` for future use.
- Modeling and Reporting: Extracting data from the `feature store`, model building, and hypothesis testing.
- Model Deployment & Monitoring: Final model deployment for generating predictive insights and supporting decision-making.

A flow chart of the process: 
![alt text](<Screenshot 2025-02-07 143305.png>)

<div style="page-break-after: always;"></div>


# 2. Preprocessing Steps

## 2.1 Project setups and tools

For this project we first created a new virtual environment using `Conda` with `Python 3.11`. For version control we are using `Git` and `Github` as the project repository. The following key libraries were used:


| Library         | Purpose                                |
|-----------------|----------------------------------------|
| **Pandas**      | Data handling, ingestion, and cleaning |
| **Numpy**       | Numerical operations                   |
| **Statsmodels** | Model building                         |
| **Streamlit**   | Frontend development and deployment    |
| **Seaborn**     | Data visualization                     |
| **Matplotlib**  | Data visualization                     |


The project had the following structure:

```bash
C:.
├───data
│   ├───raw            # Raw data before preprocessing
│   ├───intermediate   # Intermediate data used in processing
│   ├───primary        # Primary dataset/raw data
│   ├───feature_store  # Processed feature storage for reuse
│   └────lookup         # Lookup tables (e.g., country or ICD codes)
├───models              # Trained machine learning models
├───notebooks           # Jupyter notebooks for analysis, exploration and model building
│   └───__pycache__     # Python bytecode cache
├───reports             # Generated reports (e.g., Excel)
│   └───excel_report    # Excel-based reports
├───src                 # Source code for project logic
└───streamlit_app       # Streamlit application for frontend deployment
```

All analysis, exploration and model building were done in jupyter notebook, the following is a description of the function of each notebook:

- **01_raw_node.ipynb**: Handles data ingestion and data quality check with some cleaning.
- **02_intermediate_node.ipynb**.ipynb: Handles feature engineering and feature table creation.
- **03_primary.ipynb**: Performs exploratory data analysis (EDA).
- **03_Report_a_city_analysis.ipynb**: This is the full report for city-level comparison and the counts of diabetes comorbidities. The analysis extracts the feature table, performs EDA, modeling, and model diagnostics with summary results for the model.
- **04_Prediction_model_comorbidities**.ipynb: Builds and evaluates prediction models.


## 2.2 Data Ingestion

The raw layer ingests CSV files while enforcing schema validation. The goals is to ensures that each column maintains the correct data type, preventing inconsistencies in downstream processing. The data is then converted to `Parquet` format to optimize storage efficiency, ensure faster read and write operations, and preserve data types across processing stages

**_Schema Validation and Column Standardization_**

To ensure data integrity, the ingestion process applies the following:
- Explicit data type enforcement using a predefined dictionary.
- Column renaming for consistency across analysis stages.	
- Initial data validation to verify data types and structure [not done in this case].

```python
# Define the data types for each column in the dataset
dtype_dict = {
    "MEMBER_CODE": "int64",    # De-identified member ID
    "POLICY_NO": "int64",       # Policy number,  as integer
    "Age": "int64",             # Age of the member
    "BMI": "float64"            # BMI as a float
    "GENDER": "category",       # Gender as categorical 
    "CMS_Score": "int64",       # Charlson comorbidity index score, as integer
    "ICD_CODE": "category",     # ICD-10 codes as categorical
    "ICD_desc": "string",       # ICD-10 description as string
    "City": "string",           # City as string
    "CLAIM_TYPE": "category",   # Claim type is categorical
}
# Column renaming lookup table
column_lookup = {
    "MEMBER_CODE": "member_code",
    "Age": "age",
    "BMI": "bmi"
    "GENDER": "gender",
    "POLICY_NO": "policy_number",
    "CMS_Score": "cms_score",
    "ICD_CODE": "icd_code",
    "ICD_desc": "icd_description",
    "City": "city",
    "CLAIM_TYPE": "claim_type",
}
# Load the dataset with specified data types and renaming the columns
raw_data = pd.read_csv(raw_data_path, dtype=dtype_dict)

```
**Examination of the Raw data set**

An initial examination of the raw data set is done at this stage including examinig the dimensions of the data set (173772 rows and 10 columns), the data types of the columns,  inspecting the head and tails, the count of missing values per row for each column , and the number of unique values for each column. 

---
## 2.3 Data Cleaning

To ensure data consistency and integrity, a series of data cleaning steps were performed, including identifying and handling missing values, identifying and removing duplicates, and validating the data values after cleaning.

***2.3.1 Identifying and Handling Missing Data***

A summary of missing values was generated to assess completeness. The city column was identified as having 4,700 missing values, the other columns did not contain any null values. Missing values in the city column were replaced with "Unknown".

***2.3.2 Identifying and Handling Duplicates***

For duplicates the following were identified
- Two rows were identified as duplicates --> only one record was retained.
- The claim_type column showed duplication where records with an 'I' & 'O' value had  identical values across all other columns in which 34,233 out of 34,235 records with claim_type = I had an identical entry with claim_type = O (the missing two rows are probably due to not being within the sampled data extract) --> The claim type was converted to an indicator which = 1 if the claim_type = I, and 0 otherwise. Rows of the corrosponding claim_type = O were removed. 


## 2.4 Data Quality checks
To ensure the dataset maintains logical consistency and accuracy, a series of data validation checks were conducted. These checks included logical value constraints (e.g. age and bmi values), member data consistency assessments, and ICD code standardization. Any identified issues were documented for further analysis.

***2.4.1 Logical Value Checks***

Key numerical and categorical variables validated included:

***Age***

- Tested that all values fell within the range (0–120 years).
- No age value were out of the logical range

***BMI***

- Tested that  all values were within a reasonable range of (10–80). Values exceeding these may not necessarily be non-logical but may need further investigation.
- There were 20 rows with values exceeding the upper limit, of which 7 had a unique member_code and policy_number. The total records of extreme values in this case is low and we will ignore it for now.

***Gender***

- Ensured only expected categorical values ("M" or "F") were present.
- No records showed any non-logical values.

***2.4.2 Identifying Family Units Using member_code***

Some member_code values are linked to individuals of different ages and genders, suggesting that member_code is not unique to an individual but may represent a family unit instead (see example below). 

| policy_number | member_code | gender | age |
|---------------|-------------|--------|-----|
| 121           | 26730932    | M      | 78  |
| 131           | 26730932    | F      | 72  |
| 144           | 26730932    | M      | 71  |
| 176           | 26730932    | F      | 67  |

To handle thie a unique identifier for individuals was created. This is described in detail in the feature engineering section.

***2.4.3 Correct code mapping***

An ICD-10 lookup table was created for icd_code and icd_description to:

- Identify duplicate or conflicting mappings.
- Ensure consistency in ICD-10 codes across records.
- Detect missing or mismatched descriptions requiring correction.

No ICD-10 code mismatch or duplication was found.

***2.4.4 Non-Standardized City Names***

The City names were not standardazied in a specific format [e.g. RIYADH vs Ibqaiq]. To ensure consistency in geographic analysis, city names were standarized by:
- Trimming whitespace to remove unwanted spaces.
- Converting names to title case (e.g., "RIYADH" → "Riyadh").
- Creating a lookup table to store standardized city names for reusability.
 
***2.4.5 Intermediate Quality Report & Documentation***

All flagged quality issues were documented and saved in a repot which included:

- Entries with out-of-range values (e.g., unrealistic BMI or age values).
- Household-based member_code groupings for further validation.
  
A summary of member_code groupings was also saved in non_unique_member_codes.csv for reference.

<div style="page-break-after: always;"></div>

# 3. Feature Engineering & Feature Store

## 3.1 Overview
This section outlines the structured process of transforming raw data into analytical features for downstream modeling and analysis. The approach follows a layered transformation method, converting raw attributes into structured feature tables.

*Note*: Feature engineering is an iterative process that evolves based on exploratory analysis. While this report presents feature creation before descriptive analytics for clarity, the actual process involved several iterations of exploratory data analysis, transformation and testing.

## 3.2 Key Data Transformations & Feature Creation

***3.2.1 Creating Unique Identifiers***

As mentioned in the data quality section, the member code did not uniquely identify individuals. Therefore, a unique identifier was created by grouping rows using the following variables and assigning a unique number to each group:
- policy_number
- member_code
- age
- gender

A total of 18,694 unique identifiers were generated. This identifier enables differentiation of individuals within the same policy for further analysis.

***3.2.2 Numerical to Categorical Transformations***

The following numerical variables were categorized into meaningful groups:

| Feature                         | Categories                                                                                      |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| **Age Group (age_group)**       | 10-year intervals (e.g., 0-9, 10-19, 20-29, …, 80+)                                             |
| **BMI Category (bmi_cat)**      | - Underweight (<18.5) <br> - Healthy (18.5–24.9) <br> - Overweight (25–29.9) <br> - Obese (≥30) |
| **Obesity Class (obesity_cat)** | - Class 1 (30–34.9) <br> - Class 2 (35–39.9) <br> - Class 3 (≥40)                               |

These categorizations enable comparative analysis across different patient groups based on age, BMI, and obesity class.

***3.2.3 Grouping Cities with low counts into Other category***

A new feature "Major City", was created. This variable categorizes cities based on the top 5 cities by frequncy of unique patient records, while the remaining cities, including rows with an "Unknown" value, were labeled as "Other".

## 3.3 Extracting Feature Tables

In addition to the data features the following key feature tables were created [details of the method is found within the notebook `02_intermediate_node.ipynb`]:

| Feature Table                        | Description                                                                                                                                                                                                                                |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Diabetes Type Table**              | Extracts diabetes-related ICD-10 codes and calculates the total number of diabetes types reported                                                                                                                                          |
| **Diabetes Feature Table**           | Aggregates data for diabetes-related complications and calculates the total number of complications.                                                                                                                                       |
| **Comorbidity Table**                | Identifies comorbid conditions and calculates the total number of comorbidities for each patient.                                                                                                                                          |
| **Family Size Table**                | Counts the number of unique unique_id values within each policy_number and member_code pair, indicating the household or family size.                                                                                                      |
| **Unique Identifier Table**          | Combines unique identifiers with maximum BMI, BMI category, and the city with the highest occurrence per unique_id.                                                                                                                        |
| **City complications feature table** | Contains individual-level data on diabetes complications, comorbidities, and diabetes type, merged with patient demographics and city-level location information.|
| **ICD-1 Lookup Table**               | A lookup table to store for ICD-10 descriptions.                                                                                                                                                                                           |

**Diabetes Type Table**  

The Diabetes Type Table extracts, for each unique patient identifier, the unique values of ICD-10 codes that indicate a diabetes diagnosis (E09, E10, E11, E12, E13, E14). These ICD-10 codes contain the letter "E" but not the character ".". The table was created by pivoting the ICD-10 column and converting it into an indicator variable (1 for presence, 0 for absence). The values across columns were then summed to calculate the total number of diabetes types reported for each patient.

**Diabetes Complications Table**

The Diabetes Complications Table focuses on ICD-10 codes that represent diabetes-related complications. It extracts and aggregates data based on ICD-10 codes that contain both "E" and the character "." (indicating complications). Each patient’s complications were tracked and aggregated, with the total number of complications calculated by summing the individual complication indicators across columns.

**Comorbidity Table**

The Comorbidities Table identifies conditions that are comorbid with diabetes by excluding diabetes-related ICD-10 codes (those containing "E"). This table aggregates the presence of other conditions, with the total number of comorbidities for each patient calculated by summing the values across different comorbidity categories.

**City complications feature table**

The City Complications Feature Table contains individual-level data on diabetes complications, comorbidities, and diabetes type, merged with patient demographics and city-level location information. In cases where no diabetes codes are recorded for a patient, the type is assigned as Unspecified Diabetes Mellitus (E14). This is done by checking if the total number of diabetes-related ICD codes is less than 1, and if so, setting the E14 column to 1. The rationale behind this is that this dataset is a sample from the diabetes registry, where some individuals may not have explicit diabetes codes recorded but are assumed to have some form of diabetes.


## 3.4 Storing Features for Reuse

The feature tables are stored in the feature store directory as the final stage before modeling. The feature store serves as a central repository where commonly used features are stored and processed for reuse and sharing across machine learning models or teams. In practice, a feature store might be implemented as a central database with different access rights granted to data science or analytical teams across the organization. Datasets within the store will also contain metadata describing the data, the methods used to produce it, and any associated data quality checks.

```bash
C:.
├───data
│   ├───raw
│   ├───intermediate
│   ├───primary
│   └───feature_store
│           diabetes_type_feature.parquet
│           comorbidity_feature.parquet
│           diabetes_complication_feature.parquet
│           family_size_table.parquet
│           identifier_table.parquet
│           city_comp_df.parquet
├───models
├───notebooks
│   └───__pycache__
├───reports
│   └───excel_report
├───src
└───streamlit_app
```

Benefits of using a feature store:

- **Enable feature reuse and sharing**: Reusable features are available across multiple models and teams, promoting consistency and efficiency.
- **Ensure feature consistency**: Standardizes the feature data, ensuring consistency in its use across projects.
- **Maintain peak model performance**: Provides reliable, high-quality features that can help maintain the performance of models over time.
- **Enhance security and data governance**: Centralized control of features helps maintain data security and compliance with governance policies.
- **Foster collaboration between teams**: Facilitates collaboration across different teams by providing shared access to a common set of features.

note: In the current project not all the features created were necessaraly used in the downstream analysis. 

<div style="page-break-after: always;"></div>


# 4 Reports: City Level Analysis Report 

## 4.1 Introduction

Scientific evidence points to the fact that the health of individuals is increasingly associated with the built environment surrounding them. Geographic differences between cities—shaped by factors like healthcare access, socioeconomic conditions, and environmental quality—can  influence health outcomes. For individuals with chronic conditions such as diabetes, these city-level variations in resources, infrastructure, and lifestyle factors may contribute to differences in the progression of complications, with some cities offering better health outcomes than others.

In this analysis, we investigate the association between the number of diabetes complications and the city-level location of diabetic patients within Saudi Arabia, using a sample of medical claims records. By examining the geographic distribution of diabetes complications and integrating demographic information (age, sex), body measurements (BMI), and comorbidity data, we aim to explore whether living in certain cities correlates with a difference in the number of diabetes complication claims per individuals.

*objectives of the analysis is:*

- Investigate the relationship between city-level location and the frequency of diabetes complications.
- Analyze demographic and health-related factors, including age, sex, BMI, and comorbidities, and understand their role in diabetes complications at the city level.
- Provide insights for healthcare interventions or policy changes aimed at improving diabetes management and reducing complications across cities.
- Identify any significant patterns that could inform public health strategies tailored to specific geographic regions or groups.

## 4.2 Methodology

The following step were followed in the conduct of the analysis: 
1. Data Extraction: Data was extracted from the feature store, ensuring consistency and accessibility for the analysis.
2. Exploratory Data Analysis (EDA): A comprehensive analysis was performed including:
   - Summary Statistics: To provide an overview of key variables.
   - Univariate Analysis: To analyze the distribution of individual variables.
   - Bivariate Analysis: To examine relationships between pairs of variables.
   - Multivariate Analysis: To explore interactions between multiple variables.
   - Correlation Analysis: To identify significant correlations between variables.
3. Model Building: The appropriate modeling techniques are selected based on the analysis of the data.
4. Model Selection: The best-fitting model is chosen based on evaluation metrics such as AIC and BIC.
5. Model Evaluation: The model’s performance is assessed using metrics such as pseudo-R² and residual diagnostics to check for model assumptions and ensure the reliability of the model.

## 4.3 Data Extraction

Using the feature store, we extracted the dataset `city_comp_df`, which focuses on individual-level demographics and disease summary data. The dataset includes the target variable of interest, number of reported diabetes complications, the main explanatory variable, city, and other key variables such as age, BMI, number of comorbidities, type of diabetes reported, and total number of diabetes reported. Below is a description of the variables included in the dataset:

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

***4.4.2 Total complications***

On average, individuals has 1.28 reported diabetes complications of which 25% of individuals has zero reported complications. The maximum number of reported complications reached 18, indicating a wide range of reported complications count among patients. In addition around 35% of the population have no reported complication while 31% have 1 repoted complication [not shown in the table].

The distribution if total diabetes complications shows a right-skewed pattern with a significant number of individuals reporting zero complications, followed by smaller frequencies for higher counts of complications. The zero-inflated distribution indicates that many individuals report no complications, which we consider in the modeling phase, detailes are provided later.

![Complications Distribution](image-15.png)


***4.4.3 City counts***

Jeddah has the highest count of individuals, representing 35.67% of the total population, followed by Riyadh. The 'Other' city group represents 23.52% of the population. This distribution shows a disproportionate representation of cities, with Jeddah and Riyadh having a significant portion of the data.

| Category | Count | Percentage |
|----------|-------|------------|
| Jeddah   | 6668  | 35.67      |
| Riyadh   | 5273  | 28.21      |
| Alkhobar | 885   | 4.73       |
| Madina   | 771   | 4.12       |
| Makkah   | 700   | 3.74       |
| Other    | 4397  | 23.52      |


![City counts](image-13.png)


***4.4.4 Age Distribution***

The age distribution is right-skewed, with noticeable increases in counts after the ages of 50 and 60. The largest proportion of the population falls within the 60-69 age group, representing 66.60% of the population, followed by the 70-79 age group at 20.31%. The 50-59 age group accounts for 9.79%, while other age groups each represent less than 2% of the population. The average age is 64.7 years.


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

The BMI distribution is right-skewed with extreme values reaching >100. The majority of individuals (> 77%) have BMI values within the overweight and obese categories.

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

The majority of individuals have one reported diabets diagnosis, while a large proportion of individuals have no reported diabetes type. Althoug this is a data setsextracted from a diabetes registry a plausabile explanation of many records not having a diabetes type is that this is sampled from a larger data set and individuals, and information regarding the diabetes claim might have been missed.

![alt text](image-14.png)

<div style="page-break-after: always;"></div>

***4.4.7 Correlations Analysis***

Total complications is positively correlated with E10, E11, and E13 (Type I, Type II, and "Other specified diabetes mellitu", respectivly). Indicating that individuals with these diagnoses tend to have more diabetes complications.
There is a strong correlation between total complications and total DM ICD codes. City vairables show a weak correlation with total diabetes complications, with Jeddah being the only city showing a moderate positive correlation. Age showes a weak negative correlation, while BMI, and total comorbidities both showed a weak positive association with total diabetes complications. 

![alt text](image-24.png)


***4.4.8 Total Diabetes Complications by City***

The following boxplot shows the distribution of total diabetes complications across cities. There is no clear difference in median value of total complications across cities, however there is some variation in the range of values.

![boxplot city](image-30.png)

When stratifying by gender we find no visible difference in the distribution of total complications between cities. However, female patients exhibit higher variability in complication levels compared to males across cities, except in AlKhobar and Madina, where the variability between genders appears similar. this may suggest that gender may influence the degree of complications, with women having more variance in the total complications.

![boxplot city by gender](image-31.png)

When stratifying across bmi groups we observe that higher BMI (especially obesity) is linked to more total complications compared to a healthy bmi level across cities, and underweight individuals also show a slight tendency for higher complications but with less variability, though this relationship is weak.

![boxplot city by bmi cat](image-35.png)

More comorbidities are associated with higher total diabets complications across cities.

![boxplot city by comorbidities](image-33.png)

There is a very strong association between the increasing number of different diabetes diagnosis on an individual and the total diabetes comorbidities. This association is consistant across all cities. 

![boxplot city by total reported diabetes types](image-34.png)

<div style="page-break-after: always;"></div>


## 4.5 Modeling

***4.5.1 Model Slection***

The selection of the model was driven by several crucial factors:

- Nature of the Analysis: Our analysis is primarily focused on hypothesis testing, where explainability is prioritized over pure predictive performance. The goal is to understand the key factors driving diabetes complications, rather than solely optimizing predictive accuracy. This necessitated the choice of a model that would provide both insight and interpretability of the results.

- Count Nature of the Target Variable: The target variable, total diabetes complications, represents count data (i.e., the number of diabetes complications claimed per individual). This type of outcome is naturally suited for count regression models. Given the characteristics of our target variable, we considered models such as Poisson regression, which is commonly used for modeling count data.

- Model Assumptions: Poisson regression assumes equidispersion, where the mean and variance of the outcome variable are equal. However, in practice, this assumption may not always hold, especially in datasets with significant skewness or excess zeros.

**Assumption Checks:**

To test whether the assumptions of Zero-Inflation and Equidispersion hold, we performed the following checks:

1. Zero-Inflation:
   - We examined summary statistics for the number of claims and calculated the percentage of zero claims.
   - A visual inspection of the histogram of the number of claims was also conducted to assess the presence of a large number of zeros, which is indicative of zero-inflation.
2. Dispersion Statistic:
   - We calculated the variance-to-mean ratio. A value greater than 1 suggests overdispersion (where the variance exceeds the mean), indicating that Poisson regression might not be the most suitable model. The dispersion statistic for the dataset was 1.56, which is greater than 1, indicating overdispersion (see below image). This overdispersion is likely driven by the large number of zero values (many individuals reporting no complications).

To address this, we compared two potential models: the Zero-Inflated Poisson (ZIP) model and the Zero-Inflated Negative Binomial (ZINB) model. Although the ZINB model would theoretically provide a better fit due to its ability to handle overdispersion and zero inflaction, it failed to converge during model fitting. As a result, we proceeded with the Zero-Inflated Poisson (ZIP) model which is able to handle zero-inflation.

![alt text](image-36.png)


***4.5.2 Model building***

Several models were created using the zero-inflated poisson regression. The models were built with increasing complexity incorporating controlling variables [see below]. The models were then compared based on their goodness of fit, AIC, BIC, and Pseudo R² values.

The models built include:
- Model 1: A basic model with city-level effects.
- Model 2: Adds gender, age and BMI as additional predictors.
- Model 3: Adds total comorbidities as additional predictors.
- Model 4: Adds the total diabetes types claimed.

Model Comparison Criteria:
- Goodness of Fit: This is assessed by examining how well the model fits the data.
- AIC and BIC: Lower values indicate better model fit, with Model 4 performing best across both metrics.
- Pseudo R²: This metric increases with model performance, and Model 4 achieves the highest Pseudo R² value (0.107), indicating the best fit and explanatory power.

***4.5.3  Model Comparison & Diagnostics***

The following shows the summary results of the models:

**Model Comparison:**

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

*Notes*: Standard errors in parentheses [** p<.05, ***p<.01]

Model 4 explained the highest proportion of variability, with a Pseudo R² of 0.107. In addition Model 4 showed the best performance with the lowest AIC (51,987.466) and BIC (52,081.497), indicating a superior fit. Despite being the best model among the models tested, the model's overall predictive power remains modest, indicating room for improvement. This suggests that further refinement is needed, which could involve selecting a more appropriate approach or incorporating additional relevant data to better explain the variability.

**Residual Diagnostics**

The Residuals vs Fitted Values plot for model 4 shows that residuals are not randomly distributed but exhibit no clear biases or outliers.

![alt text](image-21.png)

The Q-Q plot reveals deviations from normality, particularly in the right tail, indicating poor fit for extreme values. This is likely due to the count nature of the target variable (Poisson or Negative Binomial distribution with zero inflation).

![alt text](image-22.png)

centration around zero with slight right skew, suggesting the model underpredicts extreme values. The concentration near zero may also reflect the zero-inflated nature of the data.

![alt text](image-23.png)

Overall the model diagnostic indicate that the model has difficulty fitting extreme value. 
To address this several other robust models, including those with interaction terms and clustered standard errors by policy_number, were tested, but no significant improvement in model quality was observed [results not shown here].


<div style="page-break-after: always;"></div>


## 4.6 Model Results

The following results are based on Model 4 (the full model).

***4.6.1 City-Level Effects (Main Predictor)***

The  rersults of the model shows that Alkhobar, Jeddah, and Makkah have a positive association with the total number of diabetes complications compared to the baseline city, Riyadh. The coefficients for these cities are as follows:
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


## 4.7 Discussion

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


The analysis sufferd from the following limitations: 
- **Sampling limitations**:The data has limited representation at both the individual and city levels, with an unclear and unequal sampling method. This is reflected in the population distribution, which shows overrepresentation of cities like Jeddah and Riyadh, and underrepresentation of others. The sampling method is not well-documented, potentially introducing bias into the results.
- **Individual-Level factors**: Important individual-level factors, such as socioeconomic status, education level, physical activity, and quality of care received, were not included in the model. These factors are known to be strongly associated with diabetes complications and their absence likely impacts the model's predictive power. The length of diabetes and medical history were also not considered, both of which may play significant roles in determining complication outcomes.
- **External factors**: Many external factors that could affect the probability of complications were not accounted for in the model. These include socioeconomic status, healthcare access, policy type, and neighborhood-level factors such as walkability and pollution. These missing variables likely contribute to the low explainability of the model and the unexplained variation in complications across cities. Further research is needed to incorporate these factors into future models for better interpretability.
- **Modeling limitations**: The Zero-Inflated Poisson (ZIP) model was used to account for zero-inflation and overdispersion. However, these issues led to a poor model fit, especially with extreme values being underfitted, as indicated by the residual analysis. The low variance explanation of the model suggests that the predictors used were insufficient in explaining the variation in diabetes complications.

  
## 4.9 Conclusion and Strategic Recommendations

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
  

