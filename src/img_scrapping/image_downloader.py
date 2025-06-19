import os
import random
import time

from src.img_scrapping.image_validation import is_suitable_for_face_detection
from src.img_scrapping.img_download_utils import (
    build_actor_directory,
    bytes_to_cv_image,
    get_search_variants,
    normalize_channels,
    prepare_actor_dir_name,
    read_image_data,
    save_as_jpg,
)
from src.logging_management import setup_logger

logger = setup_logger()


class ImageDownloader:
    def __init__(self, cfg_image_downloader, user_agent_manager, searcher):
        self.cfg = cfg_image_downloader
        self.user_agent_manager = user_agent_manager
        self.searcher = searcher

    def _download_image(self, url, filepath):
        response = self.searcher.fetch_html(url)
        image_bytes = read_image_data(response, chunk_size=self.cfg.chunk_size)

        is_suitable, reason = is_suitable_for_face_detection(image_bytes)
        if not is_suitable:
            logger.warning(f"Rejected image: {reason}")
            return False, "Unsuitable for face detection"

        img = bytes_to_cv_image(image_bytes)
        if img is None:
            return False, "Failed to decode image"

        img = normalize_channels(img)
        save_as_jpg(img, filepath)

        return True, "Saved successfully"

    def _download_images_batch(
        self, urls, actor_dir, base_name, total_downloaded, max_images, counter
    ):
        downloaded = 0
        for url in urls:
            if total_downloaded + downloaded >= max_images:
                break

            filename = f"{base_name}_{counter:03d}.jpg"
            filepath = os.path.join(actor_dir, filename)
            logger.info(f"Downloading [{counter}]: {filename}")

            success, reason = self._download_image(url, filepath)

            if success:
                downloaded += 1
                counter += 1
                logger.info(f"Downloaded: {filename}")
            else:
                logger.warning(f"Skipped: {reason}")

            time.sleep(random.uniform(1, 2))

        return downloaded, counter

    def download_images_for_actor(self, actor_name, num_images, output_dir):
        actor_dir = build_actor_directory(output_dir, actor_name)
        search_variants = get_search_variants(actor_name)
        actor_dir_name = prepare_actor_dir_name(actor_name)

        logger.info(f"=== Downloading images for: {actor_name} ===")
        logger.info(f"Target directory: {actor_dir}")
        logger.info(f"Search variants: {', '.join(search_variants)}")

        total_downloaded = 0
        image_counter = 1

        for variant in search_variants:
            if total_downloaded >= num_images:
                break

            logger.info(f"--- Searching: {variant} ---")
            remaining = num_images - total_downloaded
            n_variants = len(search_variants)
            n_images_suggested = max(1, remaining // n_variants)
            n_images_for_variant = min(remaining, n_images_suggested)

            image_urls = self.searcher.manage_searching_images(
                variant, n_images_for_variant * self.cfg.n_fetch_multiplier
            )
            if not image_urls:
                logger.warning(f"No results for: {variant}")
                continue

            downloaded, image_counter = self._download_images_batch(
                image_urls,
                actor_dir,
                actor_dir_name,
                total_downloaded,
                num_images,
                image_counter,
            )
            total_downloaded += downloaded

            if variant != search_variants[-1]:
                time.sleep(self.cfg.time_sleep_between_variants)

        logger.info(f"Completed! Downloaded {total_downloaded}/{num_images} images")
        logger.info(f"Location: {actor_dir}")
        return total_downloaded, actor_dir
