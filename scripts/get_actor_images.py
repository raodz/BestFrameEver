import logging
import os
import sys

from src.img_scrapping.image_scrapper import ImageScraper

DEFAULT_NUM_IMAGES = 10  # to config
DATA_DIR = "../data//actors_images"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("scraper.log")],
)
logger = logging.getLogger(__name__)


def get_user_input():
    print("=== Actor Image Scraper ===\n")

    actor_name = input("Enter actor's name: ").strip()
    if not actor_name:
        logger.error("No actor name provided!")
        return None, None

    num_images_input = input(
        f"How many images to download? (default {DEFAULT_NUM_IMAGES}): "
    ).strip()
    try:
        num_images = int(num_images_input) if num_images_input else DEFAULT_NUM_IMAGES
        if num_images <= 0:
            raise ValueError("Number must be greater than 0")
    except ValueError as e:
        logger.error(f"Invalid number of images: {e}")
        return None, None

    return actor_name, num_images


def ensure_data_directory():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created directory: {DATA_DIR}")


def main():
    """Main program function"""
    try:
        actor_name, num_images = get_user_input()
        if not actor_name:
            return

        ensure_data_directory()

        scraper = ImageScraper()

        logger.info(f"Starting download of {num_images} images for: {actor_name}")
        logger.info(f"Target directory: {DATA_DIR}")

        downloaded_count, actor_dir = scraper.download_images_for_actor(
            actor_name=actor_name, num_images=num_images, output_dir=DATA_DIR
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

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
