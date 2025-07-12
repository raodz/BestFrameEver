import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main():
    with hydra.initialize(config_path="../configs/test", version_base=None):
        cfg = hydra.compose(
            config_name="face", overrides=["hydra.output_subdir=null"]
        )  # no log

    print("📄 Loaded config:\n", OmegaConf.to_yaml(cfg))
    dataset = instantiate(cfg)
    image, target = dataset[0]
    print("🖼 Image shape:", image.shape)
    print("📦 Boxes:", target["boxes"])


if __name__ == "__main__":
    main()
