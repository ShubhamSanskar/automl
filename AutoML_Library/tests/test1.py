import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from automl import AutoML

# Load Datasets for Testing
def load_dataset(choice):
    if choice == "iris":
        data = load_iris(as_frame=True)
    elif choice == "wine":
        data = load_wine(as_frame=True)
    else:
        raise ValueError("Invalid dataset choice. Use 'iris' or 'wine'.")

    df = data.frame
    df['target'] = data.target
    return df

# Test Function
def test_automl():
    try:
        automl = AutoML()  # Initialize AutoML library

        # Load Iris Dataset for Testing
        dataset_choice = "iris"  # Change to 'wine' for Wine dataset
        df = load_dataset(dataset_choice)

        # Data Splitting
        X = df.drop("target", axis=1)
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Testing with Different Configurations
        print("\n=== Testing Linear Regression ===")
        model1 = automl.train_model("linear_regression", X_train, y_train)
        automl.evaluate_model(model1, X_test, y_test)

        print("\n=== Testing Random Forest with Custom Parameters ===")
        model2 = automl.train_model("random_forest", X_train, y_train, n_estimators=100, max_depth=5)
        automl.evaluate_model(model2, X_test, y_test)

        print("\n=== Testing SVM with Custom Parameters ===")
        model3 = automl.train_model("svm", X_train, y_train, C=1.0, kernel='linear')
        automl.evaluate_model(model3, X_test, y_test)

        print("\n=== Testing KMeans Clustering ===")
        model4 = automl.train_model("kmeans", X_train, y_train, n_clusters=3)
        automl.evaluate_model(model4, X_test, y_test)

        # Visualization Test
        print("\n=== Testing Visualization ===")
        automl.visualize_data(df, target_column="target")

        print("\n✅ All tests executed successfully.")

    except ValueError as ve:
        print(f"❌ Value Error: {ve}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

# Run the Test Code
if __name__ == "__main__":
    test_automl()
