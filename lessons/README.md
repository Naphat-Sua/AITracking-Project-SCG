# Machine-learning lessons

A hands-on collection of **machine learning and deep learning examples** in Python —
from linear regression written three ways, up to CNNs, RNN text generators, GANs and
genetic algorithms. Each folder is a self-contained lesson you can run and tweak.
These are the learning material the [AITracking app](../README.md) grew out of.

> ⚠️ **Security note:** earlier revisions of this repository contained hardcoded
> Google Maps API keys in `Generative-Algorithm/Get-Data.py`. They have been removed
> from the code, but they still exist in the git history — **revoke/rotate those keys**
> in the Google Cloud console if you have not already. The script now reads the key
> from the `GOOGLE_MAPS_API_KEY` environment variable.

> ⚠️ **Corrupted binaries removed.** Every binary file this repository tracked had
> been destroyed by a UTF-8 text round trip: bytes that are not valid UTF-8 were
> each replaced with U+FFFD, wiping out 57–70% of every file. The damage is not
> reversible, so these files were removed (they remain in git history):
>
> | File | Size | Damage |
> | --- | --- | --- |
> | `Deep-Learning/Model.h5` | 24.7 MB | 69.5% of bytes destroyed |
> | `Object-Oriented/Orient.whl` | 265 KB | 65.6% destroyed |
> | `Dataset/Stud-Classification.xlsx` | 16.9 KB | 57.3% destroyed (same data survives in the CSVs) |
> | `Dataset/Face-Classification.jpg` | 1 byte | empty placeholder |
> | `Neural-Network/Image/Logic.PNG` | 1 byte | empty placeholder |
>
> `tests/test_repo_integrity.py` fails CI if a corrupted binary is ever committed again.

## Setup

Python **3.9 – 3.12** (TensorFlow-based examples additionally need a Python
version supported by your TensorFlow build). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r lessons/requirements.txt   # numpy, pandas, sklearn, scipy, matplotlib...
pip install tensorflow                    # for the deep-learning examples
```

Each example is run from **inside its own folder**, e.g.:

```bash
cd lessons/Linear-Regression
python Regression.py
```

## What's inside

| Folder | Topic | Entry point | Extra requirements |
| --- | --- | --- | --- |
| `Activation` | Plot sigmoid / tanh / ReLU / leaky-ReLU activation functions | `Activation.py` | – |
| `Linear-Regression` | Linear & polynomial regression 7 ways (normal equation, sklearn, TF2, Keras, manual gradient descent, numpy, scipy) | `Regression.py`, `Feature.py` | TF optional |
| `Log-Regression` | Logistic regression from scratch with animated decision boundary | `Function.py` | – |
| `Neural-Network` | 2-layer network computing `X1 or X2 xor X3` (TF2), binary classification with live visualization | `Tensorflow.py`, `Class.py` | TensorFlow |
| `MNIST` | 9 classifiers on handwritten digits: nearest-neighbor, SVM, logistic regression, MLP, CNN 1D/2D, RNN, LSTM, GRU | `Test.py`, `Visual.py` | TF for examples 3-9 |
| `Principal-Component-Analysis` | PCA of digit images in 2D/3D | `PCA.py` | – |
| `Encoder-Pack` | Autoencoders for images (MNIST) and words | `Image-Encoder.py`, `Text-Encoder.py` | TensorFlow |
| `Convolutional-Network` | Neural style transfer (VGG-19) | `Artistic-Style.py` | TensorFlow, VGG-19 `.mat` weights |
| `IMGNET-Pre-Trained` | Image classification with pre-trained VGG16/VGG19/ResNet50/InceptionV3/Xception | `Image-Classifier.py` | TensorFlow, sample images |
| `Recurrent-Neural-Network` | Character-level RNN "AI writer" (articles & HTML) | `Train.py` | TensorFlow |
| `Generative-Adversarial` | GAN generating MNIST digits | `Generate-Image.py` | TensorFlow |
| `Generative-Algorithm` | Genetic algorithm: optimal road trip across Thai provinces + Google Maps distance matrix | `Get-Data.py` → `Routing.py` | `googlemaps`, `tqdm`, API key |
| `Text-Classification` | Thai text classification (article/encyclopedia/news/novel) with MLP & CNN | `thai_dataset.py` → `Text-Classify.py` | TensorFlow, `deepcut`, NECTEC corpus |
| `Dataset-Prepare` | Tour of built-in datasets in Keras and scikit-learn | `1st-Prepare.py`, `2nd-Prepare.py`, `From-MNIST.py` | TF optional |
| `Object-Oriented` | Object detection in 10 lines with `imageai` (the [AITracking app](../README.md) is the modern replacement) | `Detection.py` | `imageai`, RetinaNet weights |
| `Hash` | Kalman-filter sliding-window lane tracking utilities | `CrowndStrike.py` (library) | `filterpy` |
| `Text-Detection` | Vendored [Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN) utilities (third-party, TF1-era) | `Model.py`, `Utilities.py` | see upstream |
| `Dataset` | Small CSV datasets shared by the examples — see [`Dataset/README.md`](Dataset/README.md) | – | – |
| `Setup` | Vendored Fast R-CNN CUDA build script + upstream project notes | `PY-Setup.py` | CUDA |

Notes:

* **Nothing happens at import time.** Every script does its work inside a
  function behind `if __name__ == "__main__":`, so you can import any lesson to
  reuse its helpers without training a model, downloading a dataset or opening a
  plot window. A test enforces this.
* **Module naming** — a few helper modules use `snake_case` names
  (`route_way.py`, `thai_dataset.py`, `training_history.py`) so they can be
  imported by the scripts next to them; Python cannot import files with `-` in
  the name.
* **Vendored third-party code** (`Text-Detection/`, `Setup/PY-Setup.py`) is kept
  close to its upstream form and excluded from linting. See
  [`NOTICES.md`](../NOTICES.md) for attributions and licences.
* Scripts that need large external files (VGG weights, NECTEC corpus, RetinaNet
  weights) fail with a clear message telling you what to download and where
  to put it, instead of crashing on a hardcoded `D:/` path.

## Tests

The lesson tests live in [`../tests`](../tests) next to the app tests and run with
`pytest` from the repository root. They cover the pure-math core (activation
functions, regressions, logistic cost, genetic-algorithm operators, Kalman lane
tracking), repository integrity (no corrupted binaries) and import safety.

## Credits

Many examples are adapted from public tutorials and repositories; source links
and citations are kept in the header comments of each file, and collected in
[`NOTICES.md`](../NOTICES.md). Original learning repository by
[adminho](https://github.com/adminho/machine-learning) (see `Setup/Readme.md`).
