# Parkinson's Tremor Analytics

## Project Overview
This project analyzes tremor-related measurements in the provided Parkinson's dataset to identify meaningful patterns and relationships, especially between tremor scores and UPDRS III motor severity.

## Problem Statement
Parkinson's disease often shows variability in tremor severity across different body regions and tremor types. This project explores those patterns using a simple data analysis workflow without building a diagnosis model.

## Objectives
- Inspect the dataset and verify structure and quality
- Clean the dataset and prepare key tremor variables
- Engineer simple tremor-related features
- Explore tremor distribution and body region differences
- Examine the relationship between tremor burden and UPDRS III total
- Report evidence-based findings

## Dataset
The project uses the uploaded Parkinson's tremor CSV provided in the workspace.

## Technologies Used
- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy

## Project Workflow
1. Load dataset using the correct header row
2. Clean the data and convert relevant fields to numeric values
3. Create simple tremor features
4. Generate a few useful visualizations
5. Calculate tremor-UPDRS correlations
6. Summarize findings and conclude

## How to Install
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python parkinsons_tremor_analysis.py
```

## Key Analysis
- Distribution of tremor scores across body regions
- Comparison between rest tremor and action/postural tremor
- Relationship between total tremor burden and UPDRS III total
- Simple summary statistics for the main tremor features

## Outputs
The script saves plots in the `outputs` folder:
- tremor_distribution.png
- tremor_by_region.png
- rest_vs_action.png
- tremor_vs_updrs.png

## Limitations
- The dataset is a Parkinson's cohort and does not include a healthy control group
- This is an academic analytics project, not a medical diagnostic system
- The analysis is exploratory and based on observed relationships in the sample

## Disclaimer
This project is an academic data analytics study of tremor patterns in a Parkinson's cohort and is not a medical diagnostic system.
