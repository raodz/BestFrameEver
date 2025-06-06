import json
import logging
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from src.img_scrapping.image_validation import is_valid_image_url

logger = logging.getLogger(__name__)


class ImageSearcher:
    def __init__(self, user_agent_manager):
        self.user_agent_manager = user_agent_manager

    def search_images(self, query, num_images=10):
        logger.info(f"Searching Images: {query}")

        params = {
            "q": query,
            "first": "1",
            "count": str(min(num_images * 2, 35)),
            "qft": "+filterui:aspect-square+filterui:imagesize-medium",
        }

        url = f"https://www.bing.com/images/search?{urlencode(params)}"

        try:
            self.user_agent_manager.update_headers()
            session = self.user_agent_manager.get_session()
            response = session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            image_urls = []

            for link in soup.find_all("a", class_="iusc"):
                m_attr = link.get("m")
                if m_attr:
                    try:
                        data = json.loads(m_attr)
                        if "murl" in data:
                            url = data["murl"]
                            if is_valid_image_url(url) and url not in image_urls:
                                image_urls.append(url)
                                if len(image_urls) >= num_images:
                                    break
                    except:
                        continue

            if len(image_urls) < num_images:
                for img in soup.find_all("img"):
                    src = img.get("src") or img.get("data-src")
                    if src and is_valid_image_url(src) and src not in image_urls:
                        image_urls.append(src)
                        if len(image_urls) >= num_images:
                            break

            logger.info(f"Found {len(image_urls)} image URLs")
            return image_urls[:num_images]

        except Exception as e:
            logger.error(f"Images error: {e}")
            return []

    def manage_searching_images(self, query, num_images=10):
        all_urls = []

        logger.info("--- Trying Searching ---")
        urls = self.search_images(query, num_images)

        for url in urls:
            if url not in all_urls:
                all_urls.append(url)

        logger.info(f"Added {len(urls)} new URLs")
        logger.info(f"Total found {len(all_urls)} unique URLs")
        return all_urls[:num_images]
