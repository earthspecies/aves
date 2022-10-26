# AVES: Animal Vocalization Encoder based on Self-Supervision

![](./fig_aves.png)

AVES (Animal Vocalization Encoder based on Self-Supervision) is a self-supervised, transformer-based audio representation model for encoding animal vocalizations ("BERT for animals"). It is based on HuBERT ([Hsu et al., 2021](https://arxiv.org/abs/2106.07447)), a powerful self-supervised model for human speech, but pretrained on large-scale unannotated audio datasets ([FSD50K](https://zenodo.org/record/4060432), [AudioSet](https://research.google.com/audioset/), and [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)) which include animal sounds.

Comprehensive experiments with a suite of classification and detection tasks (from [the BEANS benchmark](https://github.com/earthspecies/beans)) have shown that AVES outperforms all the strong baselines and even the supervised "topline" models trained on annotated audio classification datasets.

See [our paper](https://arxiv.org/abs/2210.14493) for more details.

## How to use AVES

Create a conda environment by running, for example:

```
conda create -n aves python=3.8 pytorch cudatoolkit=11.3 torchvision torchaudio cudnn -c pytorch -c conda-forge
```

AVES is based on HuBERT, which is implemented in [fairseq](https://github.com/facebookresearch/fairseq), a sequence modeling toolkit developed by Meta AI. Check out [the specific commit](https://github.com/facebookresearch/fairseq/commit/eda703798dcfde11c1ee517805c27e8698285d71) of fairseq which AVES is based on, and install it via `pip`:

```
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout eda70379
pip install --editable ./
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

## Pretrained models

| Configuration      | Pretraining data            | Hours     | Link to pretrained weights   |
| ------------------ | --------------------------- | --------- | ---------------------------- |
| AVES-`core`        | FSD50k + AS (core)          | 153       | [Download](https://storage.googleapis.com/esp-public-files/aves/aves-base-core.pt) |
| AVES-`bio`         | `core` + AS/VS (animal)     | 360       | [Download](https://storage.googleapis.com/esp-public-files/aves/aves-base-bio.pt) |
| AVES-`nonbio`      | `core` + AS/VS (non-animal) | 360       | [Download](https://storage.googleapis.com/esp-public-files/aves/aves-base-nonbio.pt) |
| AVES-`all`         | `core` + AS/VS (all)        | 5054      | [Download](https://storage.googleapis.com/esp-public-files/aves/aves-base-all.pt) |
