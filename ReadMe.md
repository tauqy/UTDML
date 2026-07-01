# CS 6375 - Machine Learning Assignment 1: Linear Regression via Gradient Descent

This project implements Linear Regression using two methods: a custom Batch Gradient Descent implementation built from scratch, and Scikit-Learn's optimized Stochastic Gradient Descent (`SGDRegressor`). Both models are trained, optimized, and evaluated using the UCI Stock Portfolio Performance dataset.

## Project Structure
- `part1.py`: Custom Batch Gradient Descent class, trial logging, analytical optimization, and custom plotting.
- `part2.py`: Scikit-Learn `SGDRegressor` framework matching the exact workflow of part 1.
- `part1_custom_gd_trials_log.csv`: Optimization metrics log for the custom model.
- `part2_sklearn_trials_log.csv`: Optimization metrics log for the Scikit-Learn model.
- `README.md`: Setup, installation, and build/execution instructions.

## Prerequisites & Libraries Used
The project is built using Python 3. To run the scripts, ensure you have the following packages installed:
- `numpy` (Numerical operations and matrix manipulation)
- `pandas` (Data loading, dataframes, and preprocessing formatting)
- `matplotlib` (Graphing convergence curves and feature plots)
- `scikit-learn` (StandardScaler, train_test_split, evaluation metrics, and SGDRegressor)
- `ucimlrepo` (Direct API access to load data from the UCI Repository)

## Installation
You can install all required dependencies directly via pip:
```bash
pip install numpy pandas matplotlib scikit-learn ucimlrepo