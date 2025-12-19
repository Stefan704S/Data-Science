# Proposal — Effectiveness of Monetary Policy According to the Macroeconomic Regime

# 1. Introduction

Since 1990, the world has experienced significant economic fluctuations, including periods of growth, stagnation, and decline, forcing central banks to constantly adapt their monetary policy by raising or lowering key interest rates. However, since 2008, with weak growth and deflationary pressures, central banks around the world have introduced ever lower, even negative, interest rates.

Economic literature suggests that this decline in rates is not without consequences. Indeed, low rates may lose their effectiveness in transmitting monetary policy. According to Borio and Gambacorta (2017), when rates are very low, banks come under pressure on their margins, limiting the transmission of credit to the economy. This dynamic has led central banks around the world to use unconventional monetary policies, indicating that traditional methods are no longer sufficient to maintain price stability.

# 2. Project Structure and Methodology

This project will be divided into two parts:

# 2.1 Identification of Macroeconomic Regimes

We will identify the different regimes that Switzerland experienced between 1990 and 2024 using the K-means clustering algorithm.  
The clustering distribution is expected to yield many regimes (Elbow Curve test to identify the optimum).  
This approach avoids arbitrary human segmentation and instead classifies observations according to their degree of similarity.

# 2.2 Econometric Analysis

Once the regimes are created, we will run **multiple linear regressions** to determine the impact of the key interest rate (3-month rate) on inflation, while controlling for:

- economic growth
- exchange rate
- bank spread
- inflation

Data will come from the **Swiss National Bank**.  
The dataset will be monthly or quarterly depending on availability.

## 3. Objective

The goal is to determine whether conventional monetary policy loses its effectiveness as rates fall and approach zero.  
Success will be defined by obtaining **statistically significant and economically interpretable differences** in the coefficients across regimes.

The project combines:

- **data science**, where the algorithm determines the regimes
- **econometrics**, where the estimated betas are compared across regimes

---

# Sources

Borio, C., & Gambacorta, L. (2017). _Monetary policy and bank lending in a low interest rate environment: Diminishing effectiveness?_ (BIS Working Papers No. 612). Bank for International Settlements.
