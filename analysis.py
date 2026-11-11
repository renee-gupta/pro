"""Reference pipeline from the proposal (pandas + scikit-learn). Run: python analysis.py"""
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["annual_inc", "dti", "fico_range_low", "loan_amnt", "lti"]
GRADE_RATES = {"A": 6.5, "B": 9.5, "C": 12.5, "D": 15.5, "E": 19.0, "F": 24.0}  # illustrative


def load(path="data/loans.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"Missing file: {path}")
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])]  # drop unresolved loans
    df = df[df["annual_inc"] > 0].copy()
    df["dti"] = df["dti"].fillna(df["dti"].median())
    df["lti"] = df["loan_amnt"] / df["annual_inc"]
    df["default"] = (df["loan_status"] != "Fully Paid").astype(int)
    return df


def assign_grade(pd_value, cuts):
    grade = "A"
    for g, c in zip("BCDEF", cuts):
        if pd_value >= c:
            grade = g
    return grade


def main():
    df = load()
    print(f"{len(df)} loans, default rate {df['default'].mean():.1%}")
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df["default"], test_size=0.2, stratify=df["default"], random_state=42)
    sc = StandardScaler().fit(X_tr)
    model = LogisticRegression(class_weight="balanced", C=1e6, max_iter=1000).fit(sc.transform(X_tr), y_tr)
    prob = model.predict_proba(sc.transform(X_te))[:, 1]
    print("\nStandardised betas:")
    for name, beta in zip(FEATURES, model.coef_[0]):
        print(f"  {name:15s} {beta:+.3f}  odds ratio {pow(2.71828, beta):.2f}")
    print(f"\nROC-AUC: {roc_auc_score(y_te, prob):.3f}")
    print("Confusion matrix @0.5:\n", confusion_matrix(y_te, prob >= 0.5))
    train_prob = model.predict_proba(sc.transform(X_tr))[:, 1]
    cuts = pd.Series(train_prob).quantile([1/6, 2/6, 3/6, 4/6, 5/6]).tolist()
    grades = [assign_grade(p, cuts) for p in prob]
    print("\nActual default rate by grade:")
    print(pd.Series(y_te.values).groupby(grades).agg(["count", "mean"]).round(3))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_te, prob); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve"); plt.savefig("roc.png", dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
