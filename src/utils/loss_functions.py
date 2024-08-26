import torch
from torch.distributions import Categorical, Dirichlet
from torch.distributions.kl import _kl_dirichlet_dirichlet


class ConcordanceCorCoeff(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std

    def forward(self, prediction, ground_truth):
        mean_gt = self.mean(ground_truth, 0)
        mean_pred = self.mean(prediction, 0)
        var_gt = self.var(ground_truth, 0)
        var_pred = self.var(prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum(v_pred * v_gt) / (self.sqrt(self.sum(v_pred ** 2)) * self.sqrt(self.sum(v_gt ** 2)))
        sd_gt = self.std(ground_truth)
        sd_pred = self.std(prediction)
        numerator = 2 * cor * sd_gt * sd_pred
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
        ccc = numerator / denominator

        return 1 - ccc


def kl_divergence_edl(alpha, num_classes, device='cpu'):
    # https://github.com/dougbrion/pytorch-classification-uncertainty/

    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device='cpu'):
    # https://github.com/dougbrion/pytorch-classification-uncertainty/
    # y = y.to(device)
    alpha = alpha  # .to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence_edl(kl_alpha, num_classes, device=device)
    return A + kl_div


class WeightedEdlDigammaLoss(torch.nn.Module):
    relu_evidence = torch.nn.ReLU()

    def __init__(self, num_classes, class_weights):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.device = class_weights.device

    def forward(self, output, target, epoch_num):
        evidence = self.relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.class_weights.index_select(0, target) * edl_loss(
                torch.digamma, torch.nn.functional.one_hot(target, num_classes=self.num_classes), alpha, epoch_num,
                self.num_classes, 10, device=self.device
            )
        )
        return loss


class EdlDigammaLoss(torch.nn.Module):
    # https://github.com/dougbrion/pytorch-classification-uncertainty/

    relu_evidence = torch.nn.ReLU()

    def __init__(self, num_classes, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

    def forward(self, output, target, epoch_num):
        evidence = self.relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(
            edl_loss(
                torch.digamma, torch.nn.functional.one_hot(target, num_classes=self.num_classes), alpha, epoch_num,
                self.num_classes, 10, device=self.device
            )
        )
        return loss


def assert_no_nan_no_inf(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


# adopted from https://github.com/RylanSchaeffer/HarvardAM207-prior-networks/
def kl_divergence(model_concentrations,
                  target_concentrations,
                  mode='reverse'):
    """
    Input: Model concentrations, target concentrations parameters.
    Output: Average of the KL between the two Dirichlet.
    """
    assert torch.all(model_concentrations > 0)
    assert torch.all(target_concentrations > 0)

    target_dirichlet = Dirichlet(target_concentrations)
    model_dirichlet = Dirichlet(model_concentrations)
    kl_divergences = _kl_dirichlet_dirichlet(
        p=target_dirichlet if mode == 'forward' else model_dirichlet,
        q=model_dirichlet if mode == 'forward' else target_dirichlet)
    assert_no_nan_no_inf(kl_divergences)
    mean_kl = torch.mean(kl_divergences)
    assert_no_nan_no_inf(mean_kl)
    return mean_kl


# adopted from https://github.com/RylanSchaeffer/HarvardAM207-prior-networks/
class KlLoss(torch.nn.Module):
    def __init__(self, mode='reverse'):
        super().__init__()
        self.mode = mode
        print(f'use kl mode {mode}')

    def forward(self, model_concentrations, target_concentrations):  # (output, target)
        loss = kl_divergence(
            model_concentrations=model_concentrations,
            target_concentrations=target_concentrations,
            mode=self.mode)
        assert_no_nan_no_inf(loss)
        return loss


# adopted from https://github.com/RylanSchaeffer/HarvardAM207-prior-networks/
def weighted_kl_divergence(model_concentrations,
                           target_concentrations,
                           sample_weight,
                           mode='reverse'):
    """
    Input: Model concentrations, target concentrations parameters.
    Output: Average of the KL between the two Dirichlet.
    """
    assert torch.all(model_concentrations > 0)
    assert torch.all(target_concentrations > 0)

    target_dirichlet = Dirichlet(target_concentrations)
    model_dirichlet = Dirichlet(model_concentrations)
    kl_divergences = _kl_dirichlet_dirichlet(
        p=target_dirichlet if mode == 'forward' else model_dirichlet,
        q=model_dirichlet if mode == 'forward' else target_dirichlet)
    assert_no_nan_no_inf(kl_divergences)
    mean_kl = torch.mean(kl_divergences * sample_weight)
    assert_no_nan_no_inf(mean_kl)
    return mean_kl


# adopted from https://github.com/RylanSchaeffer/HarvardAM207-prior-networks/
class WeightedKlLoss(torch.nn.Module):
    def __init__(self, class_weights, mode='reverse'):
        super().__init__()
        self.class_weights = class_weights
        self.mode = mode
        print(f'use kl mode {mode}')

    def forward(self, model_concentrations, target_concentrations):  # (output, target)
        mean_target = target_concentrations.mean(dim=1)
        weight = target_concentrations / mean_target.unsqueeze(1)
        # weight = weight - 1/len(weight[0])
        # weight = torch.nn.functional.relu(weight)
        # weight = weight/weight.mean(dim=1).unsqueeze(1)
        weight = weight * self.class_weights
        sample_weight = weight.sum(dim=1)
        # sample_weight = sample_weight ** 2
        loss = weighted_kl_divergence(
            model_concentrations=model_concentrations,
            target_concentrations=target_concentrations,
            sample_weight=sample_weight,
            mode=self.mode)
        assert_no_nan_no_inf(loss)
        return loss
