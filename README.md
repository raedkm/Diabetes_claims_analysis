# Lean_project
A data analytics exercise to showcase the candidates technical capabilities using a healthcare sample data set related to diabetes
# Project Title

## Overview
This project involves analyzing city data to identify and handle missing data, perform feature engineering, and build predictive models. The project is structured to enable efficient collaboration and parallelization of tasks.

## Project Structure
The project is organized as follows:

```bash
C:.
├───data
│   ├───feature_store
│   ├───intermediate
│   ├───lookup
│   ├───primary
│   ├───raw
│   └───secondary
├───models
├───notebooks
│   └───__pycache__
├───reports
│   └───excel_report
├───src
└───streamlit_app
```

# Setup and Installation
1. Create a virtual environment:

```bash
conda create -n city_analysis python=3.11
conda activate city_analysis
```

2. Install the required libraries:

```bash
conda install --file environment.yml
```


# Methodology
The project follows a modular pipeline approach with the following stages:

- Data Extraction: Automated extraction from various sources like databases, APIs, and live events.
- Data Preprocessing: Data ingestion, cleaning, validation, and initial EDA.
- Feature Engineering: Creation of features stored in the feature store for future use.
- Modeling and Reporting: Teams extract data from the feature store, build models, and perform hypothesis testing.
- Model Deployment: Final model deployment for generating predictive insights and supporting decision-making.



# Notebooks
The project is run using the following notebooks:

- **01_raw_node.ipynb**: Handles data ingestion and data quality check with some cleaning.
- **02_intermediate_node.ipynb**.ipynb: Handles feature engineering and feature table creation.
- **03_primary.ipynb**: Performs exploratory data analysis (EDA).
- **03_Report_a_city_analysis.ipynb**: This is the full report for city-level comparison and the counts of diabetes comorbidities. The analysis extracts the feature table, performs EDA, modeling, and model diagnostics with summary results for the model.
- **04_Prediction_model_comorbidities**.ipynb: Builds and evaluates prediction models.



# City Analysis Project

## Overview
The Full reporet can be found by navigating to the reports directory and opening `Analysis_report.md` fo rhte markdown file and the `Analysis_report.pdf` for the pdf version. The directory hold other resources for the report.




# Streamlit app operation

The Streamlit app holds the front-end application for the Charlson Index Calculator and another app deploying the prediction model.
To run from the home directory:

```bash
cd streamlit_app
streamlit run streamlit_app.py
```
