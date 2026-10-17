# Decision Trees vs. Random Forest (Regression)

A hands-on notebook that shows **why a Random Forest generalizes better than a single
decision tree**, on a real regression task: scikit-learn's diabetes dataset (predict
disease progression one year after baseline from 10 clinical features, 442 patients).

Notebook: [`random_forest_vs_decision_tree.ipynb`](random_forest_vs_decision_tree.ipynb)
(outputs included, so it renders on GitHub).

## What it covers

1. A single decision tree **overfits** as depth grows (train R² reaches 1.00 while test
   R² collapses)
2. Random Forest: bagging plus random feature subsets reduce variance
3. 5-fold cross-validated comparison
4. Hyperparameter tuning with `RandomizedSearchCV`
5. Feature importance

## Results (seed 42)

| Model | 5-fold CV R² | Held-out test R² | Test MAE |
|---|---|---|---|
| Decision tree (full depth) | -0.145 ± 0.082 | -0.075 | 58.8 |
| Random Forest (300 trees) | **0.421 ± 0.051** | **0.474** | **43.1** |
| Random Forest (tuned) | n/a | **0.511** | **41.6** |

The fully grown tree fits the training set perfectly (R² = 1.00) but is worse than
predicting the mean on new data. The forest fixes that, and tuning adds a further gain.
Most important features: `bmi` (0.37), `s5` (0.28) and `bp` (0.13).

## Run it

```bash
pip install -r requirements.txt
jupyter notebook random_forest_vs_decision_tree.ipynb
```

## Stack

Python, scikit-learn, pandas, NumPy, Matplotlib.

## Author

**Márcio Belloni**, AI/ML Engineer.
[LinkedIn](https://www.linkedin.com/in/marcio-belloni/) · [GitHub](https://github.com/devbelloni)
