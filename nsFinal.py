import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas_ta as ta
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Load the dataset
df = pd.read_csv('1.csv', delimiter=';')

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
    'Decision Tree': {'max_depth': [5, 10, 20]},
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10]},
    'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 20]},
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
        
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies[name] = accuracy
        results[name] = accuracy
        print(f"{name} Accuracy for {retrospective} records: {accuracy * 100:.2f}%")
    
    # Neural network
    model = build_neural_network(X_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.1)
    nn_predictions = model.predict(X_test).round().astype(int).flatten()
    nn_accuracy = accuracy_score(y_test, nn_predictions)
    accuracies['Neural Network'] = nn_accuracy
    results['Neural Network'] = nn_accuracy
    print(f"Neural Network Accuracy for {retrospective} records: {nn_accuracy * 100:.2f}%")
    
    all_results.append(results)

# Save results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv('retrospective_accuracies.csv', index=False)

# Plot the results for each classifier
plt.figure(figsize=(12, 8))
for classifier in classifiers.keys():
    plt.plot(results_df['Retrospective'], results_df[classifier], label=classifier)

# Plot for Neural Network
plt.plot(results_df['Retrospective'], results_df['Neural Network'], label='Neural Network')

plt.xlabel('Retrospective (records)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Classifiers over different retrospectives')
plt.legend()
plt.savefig('combined_accuracy_retrospective.png')
plt.show()

# Save individual classifier accuracy plots
for classifier in classifiers.keys():
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Retrospective'], results_df[classifier], label=classifier)
    plt.xlabel('Retrospective (records)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy of {classifier} over different retrospectives')
    plt.legend()
    plt.savefig(f'{classifier}_accuracy_retrospective.png')
    plt.close()

# Save for Neural Network
plt.figure(figsize=(10, 6))
plt.plot(results_df['Retrospective'], results_df['Neural Network'], label='Neural Network')
plt.xlabel('Retrospective (records)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Neural Network over different retrospectives')
plt.legend()
plt.savefig('Neural_Network_accuracy_retrospective.png')
plt.close()
