# mushroom_analysis_pycharm.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def load_data():
    """Load mushroom data from UCI with validation"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]

    data = pd.read_csv(url, header=None, names=columns)
    print(f"Data loaded with {len(data)} samples")
    return data


def prepare_features(data, predictor1='odor', predictor2='gill-color'):
    """Create dummy variables with robust encoding"""
    try:
        # OneHotEncoder is more reliable than get_dummies
        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(data[[predictor1, predictor2]])
        feature_names = encoder.get_feature_names_out([predictor1, predictor2])
        X = pd.DataFrame(encoded.toarray(), columns=feature_names)
        y = data['class'].map({'e': 0, 'p': 1})  # Encode target
        return X, y
    except Exception as e:
        print(f"Encoding failed: {e}")
        return None, None


def train_model(X, y):
    """Train and evaluate Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['edible', 'poisonous'])

    return model, accuracy, report


def analyze_results(model, feature_names, accuracy, report):
    """Generate outputs and visualizations"""
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Visualization
    plt.figure(figsize=(10, 6))
    importance.head(10).plot.barh(x='feature', y='importance')
    plt.title('Top 10 Important Features for Mushroom Edibility')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    # Save results
    with open('results.txt', 'w') as f:
        f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nFeature Importance:\n")
        f.write(importance.to_string())


def main():
    print("=== Mushroom Edibility Analysis ===")

    # Load and prepare data
    data = load_data()
    X, y = prepare_features(data)

    if X is not None:
        # Train model
        model, accuracy, report = train_model(X, y)
        print(f"\nModel Accuracy: {accuracy:.2%}")

        # Generate outputs
        analyze_results(model, X.columns, accuracy, report)
        print("\nResults saved to:")
        print("- feature_importance.png")
        print("- results.txt")


if __name__ == "__main__":
    main()