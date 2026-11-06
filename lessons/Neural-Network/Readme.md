# Example: Neural network, 2 layers

## Descriptions

Two examples live in this folder:

### 1. Learning a logic function — [`Tensorflow.py`](Tensorflow.py)

A 2-layer network (5 sigmoid neurons, then 1) learns the logic
`(X1 or X2) xor X3` from the truth table in [`Layer.csv`](Layer.csv):

| X1 | X2 | X3 | X1 or X2 xor X3 |
| -- | -- | -- | --------------- |
| 0  | 0  | 0  | 0 |
| 0  | 0  | 1  | 1 |
| 0  | 1  | 0  | 1 |
| …  | …  | …  | … |

Training stops as soon as it reaches 100% accuracy, then the script runs one
test input through the learned weights in plain numpy so you can see what each
layer computed.

```bash
cd Neural-Network
python Tensorflow.py
```

The dataset path resolves relative to this folder, so there is nothing to edit.
The helpers are importable if you want to experiment:

```python
from Tensorflow import load_dataset, train, predict

df, X, Y = load_dataset()
w1, b1, w2, b2 = train(X, Y, learning_rate=0.1, seed=0)
layer1, layer2 = predict([0, 1, 1], w1, b1, w2, b2)
```

### 2. Binary classification with live visualization — [`Class.py`](Class.py)

Classifies 16 two-dimensional points into two groups, animating the decision
boundary, loss and accuracy while the network trains.

```bash
cd Neural-Network
python Class.py
```

Set `save_movie=True` in `main()` to render the animation to an mp4 instead of
showing a window (needs FFmpeg).

## Requirements

```bash
pip install tensorflow
```
