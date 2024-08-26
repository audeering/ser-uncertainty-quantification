import numpy as np
import torch


def get_torch_predict_function(model, torch_root, cfg, post_processing=None):

    if 'scaling' in cfg and cfg.scaling.type == 'temp_scaling':
        model.load_state_dict(torch.load(torch_root))
    else:
        model = model.from_pretrained(torch_root)
    model.to(cfg.device)
    model.eval()

    def predict_func(
            signal: np.ndarray,
            sampling_rate: int,
    ) -> np.ndarray:
        y = torch.from_numpy(signal).to(cfg.device)

        with torch.no_grad():
            y = model(y)
            if post_processing == 'softmax':
                y['logits'] = torch.softmax(y['logits'], dim=-1)
            elif post_processing == 'sigmoid':
                y['logits'] = torch.sigmoid(y['logits'])
            elif post_processing == 'exp':
                y['logits'] = torch.exp(y['logits']) + 1
            elif post_processing:
                raise NotImplementedError(f'postprocessing "{post_processing}", is not implemented')

        return y['logits'].squeeze().detach().cpu().numpy()

    return predict_func


def activate_dropout(model):
    # reactivate the dropout layers to analyze uncertainty through MC dropout
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.modules.dropout.Dropout):
            layer.train()

    for children in model.children():
        activate_dropout(children)


def get_torch_dropout_predict_function(model, torch_root, cfg, post_processing=None):
    model = model.from_pretrained(torch_root)
    model.to(cfg.device)
    model.eval()
    activate_dropout(model)

    def predict_func(
            signal: np.ndarray,
            sampling_rate: int,
    ) -> np.ndarray:
        y = torch.from_numpy(signal).to(cfg.device)

        with torch.no_grad():
            y = model(y)
            if post_processing == 'softmax':
                y['logits'] = torch.softmax(y['logits'], dim=-1)
            elif post_processing == 'sigmoid':
                y['logits'] = torch.sigmoid(y['logits'])
            elif post_processing:
                raise NotImplementedError(f'postprocessing "{post_processing}", is not implemented')

        return y['logits'].squeeze().detach().cpu().numpy()

    return predict_func
