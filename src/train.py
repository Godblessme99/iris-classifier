import joblib



import argparse
from pathlib import Path




import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay





def main(test_size: float, random_state: int) -> None:


    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target



    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )



    # Train model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    #Save trained model
    joblib.dump(model, "outputs/decision_tree_model.joblib")


    # Predict
    y_pred = model.predict(X_test)




    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")




    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=iris.target_names
    )




    # Ensure outputs directory exists
    Path("outputs").mkdir(exist_ok=True)

    disp.plot()
    plt.title("Iris Confusion Matrix (Decision Tree)")
    plt.savefig("outputs/confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decision Tree on the Iris dataset")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    main(test_size=args.test_size, random_state=args.random_state)