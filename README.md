# AVES: Animal Vocalization Encoder based on Self-Supervision

Update (6/5/2024): 🦜 We are excited to introduce our latest series of AVES models, called BirdAVES, specifically scaled and trained for bird sounds. [Check out the details and pretrained models here](#birdaves).

Update (7/5/2024): See our note on [batching](#warning-about-batching-in-aves-models) in AVES models.

## What is AVES?

![](./fig_aves.png)

AVES (Animal Vocalization Encoder based on Self-Supervision) is a self-supervised, transformer-based audio representation model for encoding animal vocalizations ("BERT for animals"). It is based on HuBERT ([Hsu et al., 2021](https://arxiv.org/abs/2106.07447)), a powerful self-supervised model for human speech, but pretrained on large-scale unannotated audio datasets ([FSD50K](https://zenodo.org/record/4060432), [AudioSet](https://research.google.com/audioset/), and [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)) which include animal sounds.

Comprehensive experiments with a suite of classification and detection tasks (from [the BEANS benchmark](https://github.com/earthspecies/beans)) have shown that AVES outperforms all the strong baselines and even the supervised "topline" models trained on annotated audio classification datasets.

See [our paper](https://arxiv.org/abs/2210.14493) for more details.

## How to use AVES

Create a conda environment by running, for example:

```
conda create -n aves python=3.8 pytorch cudatoolkit=11.3 torchvision torchaudio cudnn -c pytorch -c conda-forge
```

Create your working directory:

```
mkdir aves
```

Or simply clone this repository:

```
git clone https://github.com/earthspecies/aves.git
```

AVES is based on HuBERT, which is implemented in [fairseq](https://github.com/facebookresearch/fairseq), a sequence modeling toolkit developed by Meta AI. Check out [the specific commit](https://github.com/facebookresearch/fairseq/commit/eda703798dcfde11c1ee517805c27e8698285d71) of fairseq which AVES is based on, and install it via `pip`. Please note that you might encounter import issues if you install fairseq directly under your working directory. In the code below, we demonstrate installation under a sibling directory."

```
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout eda70379
pip install --editable ./
cd ../aves
```

Download the pretrained weights. See the table below for the details. We recommend the AVES-`bio` configuration, as it was the best performing model overall in our paper.

```
wget https://storage.googleapis.com/esp-public-files/aves/aves-base-bio.pt
```

You can load the model via the `fairseq.checkpoint_utils.load_model_ensemble_and_task()` method. You can implement a PyTorch classifier which uses AVES as follows. See [test_aves.py](./test_aves.py) for a working example of an AVES-based classifier. Note that AVES takes raw waveforms as input.

```python
class AvesClassifier(nn.Module):
    def __init__(self, model_path, num_classes, embeddings_dim=768, multi_label=False):

        super().__init__()

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = models[0]
        self.model.feature_extractor.requires_grad_(False)
        self.head = nn.Linear(in_features=embeddings_dim, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        out = self.model.extract_features(x)[0]
        out = out.mean(dim=1)             # mean pooling
        logits = self.head(out)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits
```


## Ported versions
The original model uses Fairseq models. We have ported the models to TorchAudio models and Onnx formats.

### TorchAudio
Download both the parameters and the model config under `TorchAudio version` in [Pretrained models](##pretrained-models).

```python
from torchaudio.models import wav2vec2_model

class AvesTorchaudioWrapper(nn.Module):

    def __init__(self, config_path, model_path):

        super().__init__()

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html

        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.feature_extractor.requires_grad_(False)

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj

    def forward(self, sig):
        # extract_feature in the sorchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]

        return out

torchaudio_model = AvesTorchaudioWrapper(config_path, model_path)
torchaudio_model.eval()

```

### Onnx
Download the parameters and the model config under `Onnx version` in [Pretrained models](##pretrained-models).
NOTE: We observed that the Onnx version of AVES-`all` could have large relative differences compared to the original version when the output values are close to zero. The TorchAudio versions don't have this problem.


```python
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: sig}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_out = ort_outs[0]
```


## Pretrained models

| Configuration      | Pretraining data            | Hours     | Link to pretrained weights   | TorchAudio version | Onnx version |
| ------------------ | --------------------------- | --------- | ---------------------------- | ------------------ | ------------ |
| AVES-`core`        | FSD50k + AS (core)          | 153       | [Download](https://storage.googleapis.com/esp-public-files/aves/aves-base-core.pt)   | [Download](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-core.torchaudio.pt) [Config](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-core.torchaudio.model_config.json) | [Download](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-core.onnx) |
| AVES-`bio`         | `core` + AS/VS (animal)     | 360       | [Download](https://storage.googleapis.com/esp-public-files/aves/aves-base-bio.pt)    | [Download](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt) [Config](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.model_config.json) | [Download](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.onnx) |
| AVES-`nonbio`      | `core` + AS/VS (non-animal) | 360       | [Download](https://storage.googleapis.com/esp-public-files/aves/aves-base-nonbio.pt) | [Download](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-nonbio.torchaudio.pt) [Config](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-nonbio.torchaudio.model_config.json) | [Download](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-nonbio.onnx) |
| AVES-`all`         | `core` + AS/VS (all)        | 5054      | [Download](https://storage.googleapis.com/esp-public-files/aves/aves-base-all.pt)    | [Download](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-all.torchaudio.pt) [Config](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-all.torchaudio.model_config.json) | [Download](https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-all.onnx) |

## Colab Notebooks
- [Supervised classification task](https://colab.research.google.com/drive/1ZmCyxSXtMVde6L_31OUnZRRWHPIxGamh?usp=sharing)
- [Clustering](https://colab.research.google.com/drive/1dtBorrZkEfsn90Mj9SETF2DFAY9sjCqe?usp=sharing)

## BirdAVES

🦜 BirdAVES is our latest series of AVES models, specifically scaled and trained for bird sounds.

* **Training Data**: In addition to the `core` configuration used for AVES, we added a large amount of bird recordings from Xeno-canto and iNaturalist for self-supervised training of BirdAVES models.
* **Model Size**: While the earlier AVES models were based on the HuBERT `base` configuration (~95M parameters), we have now successfully trained `large` models (~316M parameters) with significant performance improvements.
* **Compute**: We significantly scaled up the training compute for BirdAVES models to achieve better performance.

See the table below for detailed information and pretrained models. *BEANS avg. (all)* is the average metrics measured across all datasets from the BEANS benchmark, while *BEANS avg. (birds)* is the average metrics from datasets (cbi, dcase, enabirds, and rfcx) that include bird sounds. For more details, refer to the [BEANS paper](https://arxiv.org/abs/2210.12300).

| Configuration        | Pretraining Data               | Hours | BEANS avr. (all) | BEANS avr. (birds) | Models    |
|----------------------|--------------------------------|-------|------------------|--------------------|-----------|
| AVES-`bio` (baseline)  | `core` + AS/VS (animal)          | 360   | 0.643            | 0.419              | See above |
| BirdAVES-`biox`-base   | `bio` + xeno-canto               | 2570  | 0.678            | 0.476              |  [fairseq](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.pt) <br/> [TorchAudio](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.torchaudio.pt) ([config](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.torchaudio.model_config.json)) <br/> [ONNX](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.onnx) |
| BirdAVES-`biox`-large  | `bio` + xeno-canto               | 2570  | **0.686**        | 0.511              | [fairseq](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-large.pt) <br/> [TorchAudio](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-large.torchaudio.pt) ([config](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-large.torchaudio.model_config.json)) <br/> [ONNX](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-large.onnx) |
| BirdAVES-`bioxn`-large | `bio` + xeno-canto + iNaturalist | 3076  | 0.679            | **0.512**          | [fairseq](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-bioxn-large.pt) <br/> [TorchAudio](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-bioxn-large.torchaudio.pt) ([config](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-bioxn-large.torchaudio.model_config.json)) <br/> [ONNX](https://storage.googleapis.com/esp-public-files/birdaves/birdaves-bioxn-large.onnx) |

## Warning about batching in AVES models

Padding will affect the embeddings produced by AVES and BirdAVES models, in both the fairseq and torchaudio versions. That is, two sound signals $x$ and $x_z = \mathrm{concat}(x, \mathrm{zeros}(n))$ will give different embeddings for every frame. The `lengths` argument does not fix this issue.

This is a problem with the underlying HuBERT architecture, which could only be fixed in more recent architectures (see this [issue](https://github.com/pytorch/audio/issues/2242) and this [pull request](https://github.com/facebookresearch/fairseq/pull/3228) for more details).

This can cause confusion if you are using a model with batch size greater than 1, as the embeddings for a certain sound will depend on the batch. Therefore, we suggest pre-computing embeddings for the sounds in your dataset using `batch_size=1`. You can then pad these embeddings themselves, if using them in mini-batches for training a downstream network.
