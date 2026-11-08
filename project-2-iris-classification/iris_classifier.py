"""Classify Iris flower species with a scaled K-nearest neighbors model."""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
K_VALUES = range(1, 16)


def make_model(k: int) -> Pipeline:
    """Build a pipeline so scaling is learned only from each training fold."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )


def choose_k(features, labels) -> tuple[int, list[tuple[int, float]]]:
    """Choose K by cross-validation on the training set, never on the test set."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores_by_k = []

    for k in K_VALUES:
        scores = cross_val_score(
            make_model(k),
            features,
            labels,
            cv=folds,
            scoring="accuracy",
        )
        scores_by_k.append((k, float(scores.mean())))

    # If scores tie, prefer the smaller K value.
    best_k, _ = min(scores_by_k, key=lambda item: (-item[1], item[0]))
    return best_k, scores_by_k


def main() -> None:
    iris = load_iris()
    features = iris.data
    labels = iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names

    print("IRIS FLOWER CLASSIFICATION")
    print("=" * 29)
    print(f"Samples: {len(features)}")
    print(f"Features ({features.shape[1]}): {', '.join(feature_names)}")
    print(f"Classes ({len(target_names)}): {', '.join(target_names)}")
    print("Class counts:")
    for class_index, class_name in enumerate(target_names):
        print(f"  {class_name}: {(labels == class_index).sum()}")

    print("\nFirst five rows:")
    print(" | ".join(feature_names) + " | species")
    for row, class_index in zip(features[:5], labels[:5]):
        measurements = " | ".join(f"{value:.1f}" for value in row)
        print(f"{measurements} | {target_names[class_index]}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=labels,
    )
    print(f"\nTraining rows: {len(x_train)} ({1 - TEST_SIZE:.0%})")
    print(f"Testing rows:  {len(x_test)} ({TEST_SIZE:.0%})")

    best_k, cv_scores = choose_k(x_train, y_train)
    print("\n5-fold cross-validation accuracy by K (training data only):")
    for k, score in cv_scores:
        print(f"  K={k:2}: {score:.3f}")
    print(f"Selected K: {best_k}")

    model = make_model(best_k)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    matrix = confusion_matrix(y_test, predictions, labels=range(len(target_names)))

    print("\nHELD-OUT TEST RESULTS")
    print("=" * 21)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(f"Class order: {', '.join(target_names)}")
    print(matrix)
    print("\nPer-class report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=range(len(target_names)),
            target_names=target_names,
            zero_division=0,
        )
    )

    example = x_test[0:1]
    predicted_name = target_names[int(model.predict(example)[0])]
    actual_name = target_names[int(y_test[0])]
    print("One unseen test flower:")
    print(f"  Predicted: {predicted_name}")
    print(f"  Actual:    {actual_name}")


if __name__ == "__main__":
    main()
