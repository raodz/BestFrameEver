import logging
import os
import random
import re
import time

import cv2
import numpy as np

from src.img_scrapping.image_validation import is_suitable_for_face_detection

logger = logging.getLogger(__name__)


class ImageDownloader:
    def __init__(self, user_agent_manager, searcher):
        self.user_agent_manager = user_agent_manager
        self.searcher = searcher

    def download_image(self, url, filepath):
        try:
            self.user_agent_manager.update_headers()
            session = self.user_agent_manager.get_session()
            response = session.get(url, timeout=15, stream=True)
            response.raise_for_status()

            image_data = b""
            for chunk in response.iter_content(chunk_size=8192):
                image_data += chunk

            is_suitable, reason = is_suitable_for_face_detection(image_data)

            if not is_suitable:
                return False, f"Unsuitable: {reason}"

            image_array = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

            if img is None:
                return False, "Failed to decode image"

            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            filepath_jpg = filepath.rsplit(".", 1)[0] + ".jpg"
            cv2.imwrite(filepath_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            return True, "Saved successfully"

        except Exception as e:
            return False, f"Download error: {e}"

    def download_images_for_actor(self, actor_name, num_images=10, output_dir="data"):
        safe_actor_name = re.sub(r"[^\w\s-]", "", actor_name).strip()
        safe_actor_name = re.sub(r"[-\s]+", "_", safe_actor_name)

        actor_dir = os.path.join(output_dir, safe_actor_name)
        os.makedirs(actor_dir, exist_ok=True)

        search_variants = [
            f"{actor_name} actor portrait",
            f"{actor_name} headshot",
            f"{actor_name} face photo",
        ]

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
            images_for_variant = min(
                remaining, max(1, remaining // len(search_variants))
            )

            image_urls = self.searcher.manage_searching_images(
                variant, images_for_variant * 2
            )

            if not image_urls:
                logger.warning(f"No results for: {variant}")
                continue

            for url in image_urls:
                if total_downloaded >= num_images:
                    break

                filename = f"{safe_actor_name}_{image_counter:03d}.jpg"
                filepath = os.path.join(actor_dir, filename)

                logger.info(f"Downloading [{image_counter}]: {filename}")

                success, reason = self.download_image(url, filepath)

                if success:
                    total_downloaded += 1
                    image_counter += 1
                    logger.info(f"Downloaded: {filename}")
                else:
                    logger.warning(f"Skipped: {reason}")

                time.sleep(random.uniform(1, 2))

            if variant != search_variants[-1]:
                time.sleep(3)

        logger.info(f"Completed! Downloaded {total_downloaded}/{num_images} images")
        logger.info(f"Location: {actor_dir}")

        return total_downloaded, actor_dir
