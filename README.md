## Title

_Effectiveness of monetary policy according to the macroeconomic regime in place in Switzerland_

---

## Academic context

- **Program:** Master in Finance  
- **University:** University of Lausanne  
- **Course:** Data Science and Advanced Programming  
- **Author:** Stefan Stevanovic  
- **Project Type:** Individual Project

---

## Project description

The aim of this project is to apply data science and econometrics tools to analyze the effectiveness of monetary policy across different macroeconomic regimes in Switzerland. This project therefore has two clear objectives: to demonstrate the use of Python by examining a specific topic, in this case the effectiveness of monetary policy.

The analysis uses data from the Swiss National Bank. These data cover a monthly period from 1991 to 2024 (407 observations in total).

## How to run the project locally

1. Clone the GitHub repository

```bash
git clone https://github.com/Stefan704S/Data-Science.git
```
2. Create a virtual environment (venv)

```bash 
python -m venv .venv
```

3. Activate the virtual environment
# Windows (PowerShell)
```bash 
.\.venv\Scripts\Activate.ps1
```

# macOS / Linux
```bash 
source .venv/bin/activate
```

4. Install the required dependencies
```bash 
pip install -r requirements.txt
```

5. Run the project
```bash 
python main.py
```

---
    
## Objective

    - Apply data science tools to a real macroeconomic dataset
    - Successfully identify economic regimes based on unsupervised machine learning
    - Analyze macroeconomic regimes
    - Demonstrate proper use of Python

## Methodology

The project is divided into several parts:

    1) Data preparation
        - Data collection and entry into Excel
        - Importing data into Python
        - Standardization of variables to obtain a comparable scale between them
    
    2) Identification of regimes
        - Regimes are identified based on K-Means clustering
        - Identification of optimal K using the Elbow method
        - Verification of cluster quality and effectiveness using Anova and Silhouette test
        - Graphical visualization of clusters

    3) Econometric analysis
        - VIF test to detect multicollinearity between variables
        - OLS regression on each cluster
        - Analysis of results
        - Robustness test to compensate for heteroscedasticity

    4) Visualization
        - Elbow method
        - Two-dimensional PCA for clustering
        - Timeline for macroeconomic regimes
        - Linear regressions
        - Visualization of heteroscedasticity

---

## Data

The data are macroeconomic observations and come from the Swiss National Bank portal and the Federal Statistical Office. They cover a period from February 1991 to December 2024. The data are downloaded in raw form and implemented in Excel. The growth rate here has been calculated based on the predicted real GDP (KOF statistics).

---

# Project Structure

```text
project/
│── main.py
│── data/
│   └── raw/
│       └── data.xlsx
│        Data source/
│       └── Raw files
│── src/
│   ├── data_loader.py
│   ├── models.py
│   └── evaluation.py
│── proposal/
│── README.md
│── requirements.txt
│── environment.yml
