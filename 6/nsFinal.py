import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas_ta as ta
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Load the dataset
df = pd.read_csv('test1.csv', delimiter=';')

# Strip any extra whitespace from column names
df.columns = df.columns.str.strip()

# Ensure all column names are uppercase for consistency
df.columns = df.columns.str.upper()

def calculate_indicators(data):
    data['SMA'] = ta.sma(data['CLOSE'], length=14)
    data['EMA'] = ta.ema(data['CLOSE'], length=14)
    data['RSI'] = ta.rsi(data['CLOSE'], length=14)
    data['ATR'] = ta.atr(data['HIGH'], data['LOW'], data['CLOSE'], length=14)
    data['CCI'] = ta.cci(data['HIGH'], data['LOW'], data['CLOSE'], length=14)
    stoch = ta.stoch(data['HIGH'], data['LOW'], data['CLOSE'])
    data['STOCH_K'] = stoch['STOCHk_14_3_3']
    data['STOCH_D'] = stoch['STOCHd_14_3_3']
    macd = ta.macd(data['CLOSE'])
    data['MACD'] = macd['MACD_12_26_9']
    data['MACD_SIGNAL'] = macd['MACDs_12_26_9']
    data['MACD_HIST'] = macd['MACDh_12_26_9']
    data['WILLR'] = ta.willr(data['HIGH'], data['LOW'], data['CLOSE'], length=14)
    data['ALLIGATOR_JAW'] = ta.sma(data['CLOSE'], length=13)
    data['ALLIGATOR_TEETH'] = ta.sma(data['CLOSE'], length=8)
    data['ALLIGATOR_LIPS'] = ta.sma(data['CLOSE'], length=5)
    data['AO'] = ta.ao(data['HIGH'], data['LOW'])
    psar = ta.psar(data['HIGH'], data['LOW'], data['CLOSE'])
    data['SAR'] = psar['PSARl_0.02_0.2']
    adx = ta.adx(data['HIGH'], data['LOW'], data['CLOSE'], length=14)
    data['ADX'] = adx['ADX_14']
    return data

# Initialize classifiers with hyperparameter tuning
classifiers = {
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),  # Increase max_iter for convergence
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

params = {
    'Decision Tree': {'max_depth': [3, 5, 7, 10]},  # Reduce max depth
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10]},
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]},  # Reduce max depth and n_estimators
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9]}
}

# Function to build and compile the neural network
def build_neural_network(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

retrospectives = list(range(10, 240, 10))
all_results = []
metrics_results = []

for retrospective in retrospectives:
    print(f"Processing retrospective: {retrospective} records")
    df_retrospective = df.copy()
    df_retrospective['TARGET'] = np.where(df['CLOSE'].shift(-retrospective) > df['CLOSE'], 1, 0)
    df_retrospective = df_retrospective.iloc[:-retrospective]  # Ensure length match
    df_retrospective = calculate_indicators(df_retrospective)
    df_retrospective.dropna(inplace=True)
    
    # Define features and labels
    features = ['SMA', 'EMA', 'RSI', 'ATR', 'CCI', 'STOCH_K', 'STOCH_D', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 
                'WILLR', 'ALLIGATOR_JAW', 'ALLIGATOR_TEETH', 'ALLIGATOR_LIPS', 'AO', 'SAR', 'ADX']
    X = df_retrospective[features]
    y = df_retrospective['TARGET']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results = {'Retrospective': retrospective}
    accuracies = {}
    
    for name, clf in classifiers.items():
        if name in params:
            grid = GridSearchCV(clf, params[name], cv=5, scoring='accuracy')
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_
        else:
            clf.fit(X_train, y_train)
        
        # Cross-validation to avoid overfitting
        cross_val_acc = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean()
        
        predictions_train = clf.predict(X_train)
        predictions_test = clf.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, predictions_train)
        test_accuracy = accuracy_score(y_test, predictions_test)
        
        accuracies[f'{name} Train'] = train_accuracy
        accuracies[f'{name} Test'] = test_accuracy
        
        # Calculate additional metrics for both train and test data
        metrics_results.append({
            'Method': name,
            'Retrospective': retrospective,
            'Dataset': 'Train',
            'Accuracy': accuracy_score(y_train, predictions_train),
            'Recall': recall_score(y_train, predictions_train),
            'Precision': precision_score(y_train, predictions_train),
            'F-Score': f1_score(y_train, predictions_train)
        })
        
        metrics_results.append({
            'Method': name,
            'Retrospective': retrospective,
            'Dataset': 'Test',
            'Accuracy': accuracy_score(y_test, predictions_test),
            'Recall': recall_score(y_test, predictions_test),
            'Precision': precision_score(y_test, predictions_test),
            'F-Score': f1_score(y_test, predictions_test)
        })
        
        results[f'{name} Train'] = train_accuracy
        results[f'{name} Test'] = test_accuracy
        
        print(f"{name} Cross-Validation Accuracy for {retrospective} records: {cross_val_acc * 100:.2f}%")
        print(f"{name} Train Accuracy for {retrospective} records: {train_accuracy * 100:.2f}%")
        print(f"{name} Test Accuracy for {retrospective} records: {test_accuracy * 100:.2f}%")
    
    # Neural network
    model = build_neural_network(X_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.1)
    nn_predictions_train = model.predict(X_train).round().astype(int).flatten()
    nn_predictions_test = model.predict(X_test).round().astype(int).flatten()
    
    nn_train_accuracy = accuracy_score(y_train, nn_predictions_train)
    nn_test_accuracy = accuracy_score(y_test, nn_predictions_test)
    
    accuracies['Neural Network Train'] = nn_train_accuracy
    accuracies['Neural Network Test'] = nn_test_accuracy
    
    # Calculate additional metrics for both train and test data
    metrics_results.append({
        'Method': 'Neural Network',
        'Retrospective': retrospective,
        'Dataset': 'Train',
        'Accuracy': accuracy_score(y_train, nn_predictions_train),
        'Recall': recall_score(y_train, nn_predictions_train),
        'Precision': precision_score(y_train, nn_predictions_train),
        'F-Score': f1_score(y_train, nn_predictions_train)
    })
    
    metrics_results.append({
        'Method': 'Neural Network',
        'Retrospective': retrospective,
        'Dataset': 'Test',
        'Accuracy': accuracy_score(y_test, nn_predictions_test),
        'Recall': recall_score(y_test, nn_predictions_test),
        'Precision': precision_score(y_test, nn_predictions_test),
        'F-Score': f1_score(y_test, nn_predictions_test)
    })
    
    results['Neural Network Train'] = nn_train_accuracy
    results['Neural Network Test'] = nn_test_accuracy
    
    print(f"Neural Network Train Accuracy for {retrospective} records: {nn_train_accuracy * 100:.2f}%")
    print(f"Neural Network Test Accuracy for {retrospective} records: {nn_test_accuracy * 100:.2f}%")
    
    all_results.append(results)

# Save results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv('retrospective_accuracies.csv', index=False)

# Save metrics results to CSV
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv('metrics_results.csv', index=False)

# Plot the results for each classifier
plt.figure(figsize=(12, 8))
for classifier in classifiers.keys():
    plt.plot(results_df['Retrospective'], results_df[f'{classifier} Train'], label=f'{classifier} Train')
    plt.plot(results_df['Retrospective'], results_df[f'{classifier} Test'], label=f'{classifier} Test')

# Plot for Neural Network
plt.plot(results_df['Retrospective'], results_df['Neural Network Train'], label='Neural Network Train')
plt.plot(results_df['Retrospective'], results_df['Neural Network Test'], label='Neural Network Test')

plt.xlabel('Retrospective (records)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Classifiers over different retrospectives')
plt.legend()
plt.savefig('combined_accuracy_retrospective.png')
plt.show()

# Save individual classifier accuracy plots
for classifier in classifiers.keys():
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Retrospective'], results_df[f'{classifier} Train'], label=f'{classifier} Train')
    plt.plot(results_df['Retrospective'], results_df[f'{classifier} Test'], label=f'{classifier} Test')
    plt.xlabel('Retrospective (records)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy of {classifier} over different retrospectives')
    plt.legend()
    plt.savefig(f'{classifier}_accuracy_retrospective.png')
    plt.close()

# Save for Neural Network
plt.figure(figsize=(10, 6))
plt.plot(results_df['Retrospective'], results_df['Neural Network Train'], label='Neural Network Train')
plt.plot(results_df['Retrospective'], results_df['Neural Network Test'], label='Neural Network Test')
plt.xlabel('Retrospective (records)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Neural Network over different retrospectives')
plt.legend()
plt.savefig('Neural_Network_accuracy_retrospective.png')
plt.close()

# Select two most accurate methods for test data
top_methods = results_df.mean().sort_values(ascending=False).index[:4:2]

# Plot comparison of top methods
plt.figure(figsize=(12, 8))
for method in top_methods:
    plt.plot(results_df['Retrospective'], results_df[method], label=method)

plt.xlabel('Retrospective (records)')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Top Methods over different retrospectives')
plt.legend()
plt.savefig('top_methods_comparison.png')
plt.show()
