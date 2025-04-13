# Modeling Fatigue and Injury Risk in Athletic Movements

This repository contains the code and documentation for a research project aimed at predicting and preventing injuries in basketball players by analyzing biomechanical data and fatigue patterns using deep learning approaches.

The project combines state‑of‑the‑art LSTM architectures, comprehensive biomechanical datasets, and advanced preprocessing techniques to build predictive models that can be deployed in real‑time monitoring systems. Our models not only forecast the rate of fatigue accumulation during basketball movements but also classify joint‑specific injury risks based on subtle biomechanical signals.

## Key Objectives

- **Exhaustion Rate Prediction:** Build models to forecast fatigue progression during basketball movements.
- **Injury Risk Classification:** Develop models that analyze joint-specific biomechanical patterns to anticipate injury risks.
- **Real-time Monitoring:** Create an actionable system for use by coaches and training staff.
- **Interpretability:** Ensure that our metrics and methodologies are accessible to non‑technical stakeholders.

## Project Overview

The project leverages deep learning models – including variations of LSTM (Standard LSTM, Bidirectional LSTM, and TCN‑LSTM Hybrid) – along with comparison models such as NBEATS and Exponential Smoothing. Detailed model architectures, training methodologies, and evaluation metrics are provided in the repository. In addition, we apply sophisticated data preprocessing and feature engineering techniques on the SPL Open Biomechanics Dataset (125 basketball shooting trials), which provides joint metrics and simulated physiological data.

## Explore the Project

For a complete overview of the project, including background, methodology, models, results, and future work, please visit the [GitHub Pages site](https://ghadfield32.github.io/capstone4ds_template_IDC6940_Fatigue_Exhaustion_Time_Series_Forecasting/) generated from the `index.qmd`. This site serves as your guide through the full narrative and detailed slides of our work.

## Repository Structure

- **index.qmd:** The primary project overview and detailed documentation.
- **slides.qmd:** Presentation slides built with reveal.js for an engaging project walkthrough.
- **src/**: Contains data processing pipelines, model training scripts, and evaluation routines.
- **images/**: All images used in the documentation and presentations.
- **README.md:** This introduction file.

## Getting Started

To run or extend the project locally, please clone the repository and follow the instructions in the documentation files.

```bash
git clone https://github.com/ghadfield32/capstone4ds_template_IDC6940_Fatigue_Exhaustion_Time_Series_Forecasting.git
cd capstone4ds_template_IDC6940_Fatigue_Exhaustion_Time_Series_Forecasting
```

Further instructions for setup and running the code are available in the respective subdirectories.

## Contributions & Contact

Contributions, feedback, and suggestions are welcome. For any questions or to discuss collaboration opportunities, please contact:

- **Email:** ghadfield32@gmail.com
- **GitHub:** [@ghadfield32](https://github.com/ghadfield32)
- **LinkedIn:** [linkedin.com/in/geoffhadfield32](https://www.linkedin.com/in/geoffhadfield32)

---

Thank you for checking out our repository!
