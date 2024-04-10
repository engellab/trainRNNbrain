from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="../configs")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    task = hydra.utils.instantiate(cfg.task)

    model = hydra.utils.instantiate(cfg.model)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        num_steps=int(cfg.T / cfg.dt),
    )
    trainer.train(task, model)

if __name__ == "__main__":
    my_app()