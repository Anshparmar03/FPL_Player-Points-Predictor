# FPL Points Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains a machine learning project designed to predict player points in the Fantasy Premier League (FPL) for the 2024-2025 season, using historical data from the 2023-2024 and 2024-2025 seasons. The project trains models on 80% of the gameweeks from the 2024-2025 dataset and tests on the remaining 20%. It implements three ML algorithms: Linear Regression, Linear Discriminant Analysis (LDA), and k-Nearest Neighbors (k-NN). Performance is evaluated and compared using metrics like MSE, MAE, R² for regression, and accuracy, confusion matrices for classification.

Additionally, the trained model (Linear Regression by default) is applied to the 2023-2024 season data to predict points and compare them against actual values, generating visualizations and a comparison CSV.

This project is ideal for FPL enthusiasts, data scientists, or anyone interested in sports analytics and machine learning applications.

## Features

- **Data Preprocessing**: Handles missing values, categorical encoding (e.g., player positions, home/away status), and feature scaling.
- **Model Implementation**:
  - **Linear Regression**: For continuous point prediction.
  - **LDA**: For classifying points into categories (Low: <2, Medium: 2-5, High: >5).
  - **k-NN**: Similar classification with k=5 neighbors.
- **Evaluation**:
  - Regression metrics: MSE, MAE, R² Score.
  - Classification metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- **Cross-Season Prediction**: Uses the 2024-2025 trained model to predict and compare points for the 2023-2024 season.
- **Visualizations**: Scatter plots for actual vs. predicted points; heatmaps for confusion matrices.
- **Output**: Generates a CSV file (`points_comparison_2324.csv`) for detailed comparisons.

## Datasets

- **2024-2025 Season**: `merged_gw.csv` – Used for out-of-sample prediction and validation.
- **2023-2024 Season**: `merged_gw (23-24).csv` – Used for training and testing.

Datasets include features like expected points (xP), assists, bonuses, creativity, ICT index, minutes played, and more. Target variable: `total_points`.

**Note**: Datasets are assumed to be in CSV format with similar structures. Not included in the repo due to size; source from official FPL APIs or community scrapers.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fpl-points-predictor.git
   cd fpl-points-predictor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   `requirements.txt` example:
   ```
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   ```

3. Place datasets (`merged_gw.csv` and `merged_gw (23-24).csv`) in the root directory.

## Usage

Run the main script:
```
python fpl_points_prediction.py
```

- **Output**:
  - Console logs with model performance metrics.
  - Confusion matrix visualizations for LDA and k-NN.
  - Scatter plot for 2023-2024 actual vs. predicted points.
  - CSV file: `points_comparison_2324.csv`.

Customize:
- Adjust features in `preprocess_data` function.
- Change classification thresholds in `categorize_points`.
- Switch to LDA/k-NN for 2023-2024 predictions if needed.

## Models and Performance

- **Training Split**: 80/20 based on gameweeks to prevent leakage.
- **Example Performance** (results vary by run):
  - Linear Regression: MSE ~5-10, MAE ~1-2, R² ~0.6-0.8.
  - LDA/k-NN: Accuracy ~70-85%, with confusion matrices showing strengths in medium-point classification.

Observations from the project:
- Linear Regression excels at precise predictions but may over/under-estimate extremes.
- LDA and k-NN are better for categorizing player performance, useful for FPL team selection strategies.

## Contributing

Contributions are welcome! Fork the repo, create a branch, and submit a pull request. Focus on:
- Adding more models (e.g., Random Forest, XGBoost).
- Enhancing visualizations.
- Supporting real-time FPL API integration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Fantasy Premier League data analysis communities.
- Built with scikit-learn for ML and pandas for data handling.

For questions, open an issue or contact [your.email@example.com]. Happy FPL season! ⚽
