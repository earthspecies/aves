import fairseq
import torch
import torch.nn as nn


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


# Initialize an AVES classifier with 10 target classes
model = AvesClassifier(
    model_path='./aves-base-bio.pt',
    num_classes=10)

# Create a 1-second random sound
waveform = torch.rand((16_000))
x = waveform.unsqueeze(0)
y = torch.tensor([0])

# Run the forward pass
loss, logits = model(x, y)
