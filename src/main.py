import hydra
from omegaconf import DictConfig, OmegaConf

from evaluation.evaluation_main import evaluate
import train
import w2v2_cat
from utils.helper_functions import set_device


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_device(cfg)
    print(cfg)
    model = w2v2_cat.ModelCategorical
    path = train.train(model, cfg)
    print(f'Training finished, model available under {path}')

    report = evaluate(model, path, cfg)


if __name__ == "__main__":
    main()
