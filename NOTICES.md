# Third-party code and attributions

This repository is a teaching collection, and several files are copied or
adapted from other open-source projects. Their original licences and copyright
notices apply to those files and are preserved in the file headers.

## Vendored source files

| Path | Upstream project | Licence | Copyright |
| --- | --- | --- | --- |
| `Text-Detection/Utilities.py` | [Mask R-CNN](https://github.com/matterport/Mask_RCNN) | MIT | © 2017 Matterport, Inc. — written by Waleed Abdulla |
| `Text-Detection/Model.py` | SPCNet / Mask R-CNN derived text-detection network | see upstream | – |
| `Setup/PY-Setup.py` | [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn) | MIT | © 2015 Microsoft — written by Ross Girshick |
| `Generative-Algorithm/Show.template` | [Randal S. Olson's optimal road trip](http://www.randalolson.com/2015/03/10/computing-the-optimal-road-trip-across-europe/) | see upstream | © Randal S. Olson |

These files are excluded from this project's linting (see `pyproject.toml`) so
they stay close to their upstream form and remain easy to diff against it.

## Adapted examples

The following lessons are adaptations rather than verbatim copies. Source links
are kept in each file's header comments:

- `Generative-Algorithm/route_way.py`, `Get-Data.py` — adapted from Randal S.
  Olson's optimal road trip notebook.
- `Recurrent-Neural-Network/Writer.py` — adapted from the Keras
  `lstm_text_generation.py` example (François Chollet).
- `Text-Classification/Text-Classify.py` — adapted from the Keras
  `reuters_mlp.py` example.
- `Generative-Adversarial/Generate-Image.py` — adapted from
  [osh/KerasGAN](https://github.com/osh/KerasGAN); paper:
  [Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661).
- `Convolutional-Network/Artistic-Style.py` — adapted from
  [Anish Athalye's Neural Style](https://github.com/anishathalye/neural-style);
  paper: [Gatys et al., 2015](https://arxiv.org/pdf/1508.06576v2.pdf).
- `Hash/CrowndStrike.py` — Kalman-filter lane-tracking windows, adapted from a
  highway lane detection project.
- `Neural-Network/Class.py` — adapted from Karpathy's ConvNetJS 2D
  classification demo and the scikit-learn SVM iris example.
- `IMGNET-Pre-Trained/Image-Classifier.py` — uses `keras.applications`
  (see [keras.io/applications](https://keras.io/applications)).
- `MNIST/`, `Principal-Component-Analysis/` — built on scikit-learn and Keras
  documentation examples.
- The original learning repository this collection grew from is
  [adminho/machine-learning](https://github.com/adminho/machine-learning);
  its README is preserved at `Setup/Readme.md`.

## Datasets

| Dataset | Source |
| --- | --- |
| Food truck (`Linear-Regression/Dataset.csv`) | Andrew Ng's [Machine Learning course](https://www.coursera.org/learn/machine-learning) |
| Exam scores (`Dataset/Norm-Classification.csv`) | Andrew Ng's Machine Learning course |
| Thailand population history (`Linear-Regression/Population-Linear.csv`) | [countrymeters.info](http://countrymeters.info/en/Thailand) |
| Thai word frequency (`Text-Classification/Frequency.csv`) | [Thai National Corpus](http://www.arts.chula.ac.th/~ling/TNC/), Chulalongkorn University |
| Thai text corpora (article / encyclopedia / news / novel) | [NECTEC corpus](https://www.nectec.or.th/corpus/index.php?league=pm) (downloaded separately) |
| Digits, California housing, CIFAR, MNIST, IMDB, Reuters | bundled with scikit-learn / Keras |

## Licence for this repository

This project does **not** currently declare a licence of its own. Without one,
default copyright applies and others have no explicit permission to reuse the
original (non-vendored) code here.

If you own this repository and want it to be reusable, add a `LICENSE` file —
MIT is the common choice for a teaching collection like this one, and it is
already compatible with the vendored MIT-licensed files listed above.
Choosing a licence is the owner's decision, so none has been added here.
