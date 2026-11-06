# AITracking-Project-SCG

A hands-on collection of **machine learning and deep learning examples** in Python —
from linear regression written three ways, up to CNNs, RNN text generators, GANs and
genetic algorithms. Each folder is a self-contained lesson you can run and tweak.

> ⚠️ **Security note:** earlier revisions of this repository contained hardcoded
> Google Maps API keys in `Generative-Algorithm/Get-Data.py`. They have been removed
> from the code, but they still exist in the git history — **revoke/rotate those keys**
> in the Google Cloud console if you have not already. The script now reads the key
> from the `GOOGLE_MAPS_API_KEY` environment variable.

> ⚠️ **Corrupted binaries removed.** Every binary file this repository tracked had
> been destroyed by a UTF-8 text round trip: bytes that are not valid UTF-8 were
> each replaced with U+FFFD, wiping out 57–70% of every file. Text files were
> unaffected. The damage is not reversible, so these five files were removed:
>
> | File | Size | Damage |
> | --- | --- | --- |
> | `Deep-Learning/Model.h5` | 24.7 MB | 69.5% of bytes destroyed |
> | `Object-Oriented/Orient.whl` | 265 KB | 65.6% destroyed |
> | `Dataset/Stud-Classification.xlsx` | 16.9 KB | 57.3% destroyed (same data survives in the CSVs) |
> | `Dataset/Face-Classification.jpg` | 1 byte | empty placeholder |
> | `Neural-Network/Image/Logic.PNG` | 1 byte | empty placeholder |
>
> They remain in git history if you want to inspect them. Re-add good copies
> from their original sources if you need them — and transfer binaries in
> binary mode, never through a text pipeline. `tests/test_repo_integrity.py`
> now fails CI if a corrupted binary is ever committed again.

## Setup

Python **3.9 – 3.12** is supported and tested in CI (TensorFlow-based examples
additionally need a Python version supported by your TensorFlow build).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt              # core: numpy, pandas, sklearn, scipy, matplotlib...
pip install tensorflow                       # for the deep-learning examples
pip install -r requirements-dev.txt          # for development: pytest + ruff
```

Each example is run from **inside its own folder**, e.g.:

```bash
cd Linear-Regression
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
| `Object-Oriented` | Object detection in 10 lines with `imageai` | `Detection.py` | `imageai`, RetinaNet weights |
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
  [`NOTICES.md`](NOTICES.md) for attributions and licences.
* Scripts that need large external files (VGG weights, NECTEC corpus, RetinaNet
  weights) fail with a clear message telling you what to download and where
  to put it, instead of crashing on a hardcoded `D:/` path.

## Development

```bash
ruff check .        # lint
pytest              # run the unit tests (tests/ folder)
```

Both run automatically in CI (GitHub Actions) on every push and pull request,
with tests executed against Python 3.9, 3.10, 3.11 and 3.12.

The suite covers three things and finishes in a few seconds, without needing
TensorFlow:

1. **The pure-math core** — activation functions, encode/decode helpers, the
   normal-equation and gradient-descent regressions, the logistic cost
   function, the genetic-algorithm operators, and the Kalman lane-tracking
   windows.
2. **Repository integrity** — every tracked binary must have a valid magic
   number and contain no U+FFFD replacement characters, so the corruption
   described above cannot silently return.
3. **Import safety** — every lesson module must import silently, create no
   matplotlib figures, and keep its demo body behind a `__main__` guard
   (checked by parsing the AST, so no heavy dependencies are needed).

## Licence

This project does not declare a licence of its own; see
[`NOTICES.md`](NOTICES.md) for what that means and for the licences of the
vendored third-party files.

## Credits

Many examples are adapted from public tutorials and repositories; source links
and citations are kept in the header comments of each file, and collected in
[`NOTICES.md`](NOTICES.md). Original learning repository by
[adminho](https://github.com/adminho/machine-learning) (see `Setup/Readme.md`).
