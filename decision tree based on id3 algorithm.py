import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

def main():
    training_data = pd.read_csv('C:\\Users\\ajayk\\OneDrive\\Documents\\finds and ce.csv')

    # One-hot encode categorical features
    training_data_encoded = pd.get_dummies(training_data)

    X_train = training_data_encoded.iloc[:, :-1]
    y_train = training_data_encoded.iloc[:, -1]

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    new_sample = pd.DataFrame([['Oval', 'Small', 'Dark', 'Irregular', 'Thin']], columns=['Shape', 'Size', 'Color', 'Surface', 'Thickness'])
    new_sample_encoded = pd.get_dummies(new_sample)  # One-hot encode the new sample

    # Ensure that the feature names match those seen during model fitting
    missing_cols = set(X_train.columns) - set(new_sample_encoded.columns)
    for col in missing_cols:
        new_sample_encoded[col] = 0

    # Reorder columns to match the order seen during model fitting
    new_sample_encoded = new_sample_encoded[X_train.columns]

    prediction = clf.predict(new_sample_encoded)
    print("Prediction for the new sample:", prediction[0])

if __name__ == "__main__":
    main()
