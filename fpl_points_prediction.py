import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
data_2425 = pd.read_csv('merged_gw.csv')
data_2324 = pd.read_csv('merged_gw (23-24).csv')

# Preprocess the data
def preprocess_data(df):
    # Select relevant features
    features = [
        'xP', 'assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 
        'expected_assists', 'expected_goal_involvements', 'expected_goals', 
        'expected_goals_conceded', 'goals_conceded', 'goals_scored', 
        'ict_index', 'influence', 'minutes', 'own_goals', 
        'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 
        'threat', 'transfers_balance', 'transfers_in', 'transfers_out', 
        'value', 'yellow_cards'
    ]
    target = 'total_points'

    # Handle missing values
    df = df.fillna(0)

    # Convert categorical variables
    df['was_home'] = df['was_home'].astype(int)
    position_mapping = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3, 'AM': 4}
    df['position'] = df['position'].map(position_mapping).fillna(-1)  # Handle unknown positions

    # Ensure all selected features exist in the dataframe
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df[target]

    return X, y, features

# Preprocess 2024-2025 data
X_2425, y_2425, selected_features = preprocess_data(data_2425)

# Split gameweeks into training (80%) and testing (20%)
gameweeks = data_2425['GW'].unique()
train_gw, test_gw = train_test_split(gameweeks, test_size=0.2, random_state=42)

train_data = data_2425[data_2425['GW'].isin(train_gw)]
test_data = data_2425[data_2425['GW'].isin(test_gw)]

X_train = train_data[selected_features]
y_train = train_data['total_points']
X_test = test_data[selected_features]
y_test = test_data['total_points']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Convert regression output to classification for LDA and k-NN
def categorize_points(points):
    if points < 2:
        return 0  # Low
    elif points <= 5:
        return 1  # Medium
    else:
        return 2  # High

y_train_class = y_train.apply(categorize_points)
y_test_class = y_test.apply(categorize_points)

# Linear Discriminant Analysis
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_scaled, y_train_class)
lda_pred = lda_model.predict(X_test_scaled)

# k-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train_class)
knn_pred = knn_model.predict(X_test_scaled)

# Evaluate models
def evaluate_regression(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Regression Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    return {'MSE': mse, 'MAE': mae, 'R2': r2}

def evaluate_classification(y_true, y_pred, model_name):
    print(f"\n{model_name} Classification Performance:")
    print(classification_report(y_true, y_pred, target_names=['Low', 'Medium', 'High']))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return cm

# Evaluate Linear Regression
lr_metrics = evaluate_regression(y_test, lr_pred, "Linear Regression")

# Evaluate LDA
lda_cm = evaluate_classification(y_test_class, lda_pred, "LDA")

# Evaluate k-NN
knn_cm = evaluate_classification(y_test_class, knn_pred, "k-NN")

# Compare model performance
print("\nModel Performance Comparison:")
print("Linear Regression Metrics:", lr_metrics)
from sklearn.metrics import accuracy_score
lda_accuracy = accuracy_score(y_test_class, lda_pred)
knn_accuracy = accuracy_score(y_test_class, knn_pred)
print(f"LDA Accuracy: {lda_accuracy:.2f}")
print(f"k-NN Accuracy: {knn_accuracy:.2f}")

# Observations
print("\nObservations:")
print("- Linear Regression provides a continuous prediction of points, suitable for precise point estimation.")
print("- LDA and k-NN classify points into categories, which may be useful for grouping players by performance level.")
print("- Linear Regression's MSE and MAE indicate the average error in point predictions, while R² shows the proportion of variance explained.")
print("- LDA and k-NN's confusion matrices reveal how well they classify players into low, medium, and high point categories.")
print("- If precise point prediction is the goal, Linear Regression is likely more appropriate. If categorization is preferred, compare LDA and k-NN accuracies.")

# Part 2: Predict points for 2023-2024 season
# Preprocess 2023-2024 data
X_2324, y_2324, _ = preprocess_data(data_2324)

# Ensure features match the training data
X_2324 = X_2324[selected_features]

# Standardize features
X_2324_scaled = scaler.transform(X_2324)

# Predict using Linear Regression (assumed best for continuous output)
predictions_2324 = lr_model.predict(X_2324_scaled)

# Create a DataFrame to compare actual vs predicted points
comparison_df = pd.DataFrame({
    'Player': data_2324['name'],
    'Actual_Points': y_2324,
    'Predicted_Points': predictions_2324
})

# Calculate performance metrics for 2023-2024 predictions
mse_2324 = mean_squared_error(y_2324, predictions_2324)
mae_2324 = mean_absolute_error(y_2324, predictions_2324)
r2_2324 = r2_score(y_2324, predictions_2324)

print("\n2023-2024 Season Prediction Performance:")
print(f"MSE: {mse_2324:.2f}")
print(f"MAE: {mae_2324:.2f}")
print(f"R² Score: {r2_2324:.2f}")

# Save comparison to CSV
comparison_df.to_csv('points_comparison_2324.csv', index=False)

# Plot actual vs predicted points
plt.figure(figsize=(10, 6))
plt.scatter(comparison_df['Actual_Points'], comparison_df['Predicted_Points'], alpha=0.5)
plt.plot([comparison_df['Actual_Points'].min(), comparison_df['Actual_Points'].max()], 
         [comparison_df['Actual_Points'].min(), comparison_df['Actual_Points'].max()], 
         'r--', lw=2)
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Actual vs Predicted Points (2023-2024 Season)')
plt.show()

print("\nComparison file 'points_comparison_2324.csv' has been generated.")