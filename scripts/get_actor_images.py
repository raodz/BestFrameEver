import os
import sys

import hydra
from omegaconf import DictConfig

from src.img_scrapping.image_scrapper import ImageScraper
from src.logging_management import setup_logger
from src.utils.paths import ACTORS_IMAGES_PATH

logger = setup_logger()


@hydra.main(
    version_base=None, config_path="../configs/img_scrapping", config_name="default"
)
def main(cfg: DictConfig):
    """Main program function"""
    actor_name = cfg.get_actor_images.actor_name
    num_images = cfg.get_actor_images.num_images
    if not actor_name:
        return

    data_dir = os.path.join(f"../{ACTORS_IMAGES_PATH}")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created directory: {data_dir}")

    scraper = ImageScraper(cfg)

    logger.info(f"Starting download of {num_images} images for: {actor_name}")
    logger.info(f"Target directory: {data_dir}")

    downloaded_count, actor_dir = scraper.download_images_for_actor(
        actor_name=actor_name, num_images=num_images, output_dir=data_dir
    )

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY:")
    logger.info(f"   Actor: {actor_name}")
    logger.info(f"   Downloaded: {downloaded_count}/{num_images} images")
    logger.info(f"   Location: {actor_dir}")

    if downloaded_count < num_images:
        missing = num_images - downloaded_count
        logger.warning(f"   Missing: {missing} images")
        logger.info("   Hint: Try again in a few minutes")
    else:
        logger.info("   All images downloaded successfully!")

    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
