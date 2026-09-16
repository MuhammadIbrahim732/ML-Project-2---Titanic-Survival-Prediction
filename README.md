# 🚢 Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using classic classification algorithms. The project covers the full ML pipeline — data cleaning, preprocessing, model training, evaluation, and cross-validation — comparing five different classifiers on the same dataset.

## 📊 Dataset

The project uses the built-in **Titanic dataset** from Seaborn (`sns.load_dataset("titanic")`), which contains passenger information such as class, sex, age, fare, and survival status.

## 🛠️ Workflow

1. **Import Libraries** — NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
2. **Load Dataset** — Titanic dataset via Seaborn
3. **Exploratory Data Analysis (EDA)**
   - Dropped redundant/derived columns (`class`, `who`, `adult_male`, `deck`, `embark_town`, `alive`)
   - Checked and handled missing values (`age` filled with mean, rows with missing `embarked` dropped)
   - Checked for duplicate records
   - Encoded categorical features (`sex`, `embarked`) using `LabelEncoder`
4. **Train-Test Split** — 80/20 split (`random_state=42`)
5. **Model Training & Evaluation** — Five classifiers trained and evaluated on accuracy, confusion matrix, and classification report:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Naive Bayes (GaussianNB)
   - Decision Tree
   - Support Vector Machine (SVM, RBF kernel)
6. **K-Fold Cross Validation** — 5-fold cross-validation to compare model generalization

## 📈 Results

### Test Set Accuracy

| Model | Accuracy |
|---|---|
| Logistic Regression | 80.3% |
| KNN | 79.8% |
| Naive Bayes | 79.8% |
| Decision Tree | 74.7% |
| **SVM (RBF)** | **81.5%** |

### 5-Fold Cross-Validation Mean Accuracy

| Model | CV Accuracy |
|---|---|
| Logistic Regression | 67.1% |
| KNN | 70.6% |
| **Naive Bayes** | **78.6%** |
| Decision Tree | 77.8% |
| SVM | 67.1% |

> Naive Bayes and Decision Tree show the most consistent performance across cross-validation folds, while SVM and Logistic Regression scores drop in CV compared to the single test split — suggesting some sensitivity to the specific train/test partition and a case for further tuning (e.g. feature scaling for Logistic Regression, kernel/`C` tuning for SVM).

## 🧰 Tech Stack

- **Python 3**
- **Pandas** & **NumPy** — data manipulation
- **Matplotlib** & **Seaborn** — visualization and dataset loading
- **Scikit-learn** — preprocessing, models, and evaluation metrics

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run

1. Clone this repository
2. Open the notebook:

```bash
jupyter notebook ML_Project-1.ipynb
```

3. Run all cells sequentially

## 📁 Project Structure

```
.
├── ML_Project-1.ipynb   # Main notebook: EDA, preprocessing, model training & evaluation
└── README.md
```

## 🔮 Future Improvements

- Scale features for Logistic Regression and SVM to improve CV stability
- Hyperparameter tuning (GridSearchCV) for KNN, Decision Tree, and SVM
- Feature engineering (e.g. family size from `sibsp`/`parch`, title extraction)
- Try ensemble methods (Random Forest, Gradient Boosting, XGBoost)

## 👤 Author

**Muhammad Ibrahim**

- 📧 Email: [mibrahim.seng@gmail.com](mailto:mibrahim.seng@gmail.com)
- 💼 LinkedIn: [muhammad-ibrahim-python](https://www.linkedin.com/in/muhammad-ibrahim-python)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
