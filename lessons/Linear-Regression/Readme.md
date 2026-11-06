# Linear regression

1) Example predicts food truck profit
2) Example predicts Thailand population history (polynomial features)
3) Example predicts California house prices (many features)

## How to run

```bash
cd Linear-Regression
python Regression.py   # 7 methods compared on each dataset
python Feature.py      # single-feature walkthrough (7 methods)
```

# Datasets

| Example | Dataset file | Cites |
| --- | --- | -- |
| Food truck | [Dataset.csv](Dataset.csv) (columns: `Input`, `Output`) | [course online](https://www.coursera.org/learn/machine-learning) taught by Andrew Ng |
| Thailand population history | [Population-Linear.csv](Population-Linear.csv) (columns: `Year`, `Population`) | [http://countrymeters.info/en/Thailand](http://countrymeters.info/en/Thailand) |
| California housing | loaded with `sklearn.datasets.fetch_california_housing()` (sklearn removed `load_boston`) | [California housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) |

`Income-Linear.csv` (average income per month per household of Thailand,
B.E 41-58) is currently empty — re-download it from
[data.go.th](https://data.go.th) if you want to run that example.
