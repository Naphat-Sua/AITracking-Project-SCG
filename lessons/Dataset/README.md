# Shared datasets

Small CSV files used by (or available to) the examples in this repository.
All are tiny enough to read in full before using them.

| File | Rows | Columns | Used by | Notes |
| --- | --- | --- | --- | --- |
| `Norm-Classification.csv` | 100 | `X`, `Y`, `Label` | [Log-Regression](../Log-Regression) | Two exam scores per student and a 0/1 admission label. The classic logistic-regression dataset from Andrew Ng's ML course. `Function.py` renames the columns to `X1`, `X2`, `Y`. |
| `Fore-Classification.csv` | 13 | `Temperature`, `Humidity`, `Fires` | – | Toy binary classification: does a fire start? `Fires` is `Yes`/`No` (encode before use). |
| `Dirt-Classification.csv` | 32 | `Duration`, `Date`, `Pulse`, `Maxpulse`, `Calories` | – | Deliberately "dirty" workout log for data-cleaning practice: `Date` values are quoted (`'2020/12/01'`) so they parse as strings, and 3 cells are empty. |
| `Pred-Classification.csv` | 23 | `years`, `Wildebeest`, `Zebra` | – | Two animal populations over 23 years. Suitable for regression / time-series or predator-prey practice. |
| `Seco-Classification.csv` | 20 | `เลขประจำตัว`, `เพศ`, `Cs`, `height` | – | Student records: ID, sex (1/2), a course score, and height in cm. Thai column headers. |
| `Scor-Classification.csv` | 20 | `เลขประจำตัว`, `Science1` | – | Student ID and a science score, joinable with `Seco-Classification.csv` on `เลขประจำตัว`. |
| `Stud-Classification.csv` | 19 (+1) | *none* | – | ⚠️ **Same data as `Seco-Classification.csv` but with no header row.** See the warning below. |

## ⚠️ `Stud-Classification.csv` has no header row

Reading it the usual way silently swallows the first student, because pandas
treats that row as the column names:

```python
pd.read_csv("Stud-Classification.csv")
# columns: ['5884', '2', '18', '158']   <- that was a data row!
# rows:    19                            <- one student lost
```

Read it explicitly instead:

```python
pd.read_csv("Stud-Classification.csv",
            header=None,
            names=["เลขประจำตัว", "เพศ", "Cs", "height"])
# 20 rows, matching Seco-Classification.csv
```

Prefer `Seco-Classification.csv`, which carries the same 20 records with proper
headers.

## Removed binary files

`Face-Classification.jpg` and `Stud-Classification.xlsx` used to live here but
were unrecoverably corrupted (see the note in the [root README](../README.md)),
so they have been removed. `Stud-Classification.xlsx` held the same data as the
CSVs above, so nothing was lost.
