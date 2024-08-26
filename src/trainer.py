import transformers
import torch


class Trainer(transformers.Trainer):

    def __init__(
            self,
            criterion: torch.nn.Module,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.criterion = criterion

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
    ):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs[0]
        # logits = torch.nn.functional.softmax(logits)
        loss = self.criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss


class EdlTrainer(transformers.Trainer):

    def __init__(
            self,
            criterion: torch.nn.Module,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.criterion = criterion

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
    ):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs[0]
        loss = self.criterion(logits, labels, int(self.state.epoch))

        return (loss, outputs) if return_outputs else loss


def onehot_with_ignore_label(labels, num_class, ignore_label):
    # https://discuss.pytorch.org/t/how-could-i-create-one-hot-tensor-while-ignoring-some-label-index/40987/5
    dummy_label = num_class + 1
    mask = labels == ignore_label
    modified_labels = labels.clone()
    modified_labels[mask] = num_class
    # One-hot encode the modified labels
    one_hot_labels = torch.nn.functional.one_hot(modified_labels, num_classes=dummy_label)
    # Remove the last row in the one-hot encoding
    one_hot_labels = one_hot_labels[:, :-1]
    return one_hot_labels


class KlTrainer(transformers.Trainer):

    def __init__(
            self,
            criterion: torch.nn.Module,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.criterion = criterion

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
    ):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs[0]
        target_concentration = torch.ones(logits.shape).to(logits.device)
        lab = onehot_with_ignore_label(labels, num_class=logits.shape[-1], ignore_label=-1) * 100

        target_concentration = target_concentration + lab

        concentrations = torch.exp(logits) + 1
        loss = self.criterion(concentrations, target_concentration)

        return (loss, outputs) if return_outputs else loss
