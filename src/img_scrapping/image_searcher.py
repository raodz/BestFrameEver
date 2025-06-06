import json
from urllib.parse import urlencode

from bs4 import BeautifulSoup
from omegaconf import OmegaConf

from src.img_scrapping.image_validation import is_valid_image_url
from src.logging_management import setup_logger

logger = setup_logger()


class ImageSearcher:
    def __init__(self, cfg, user_agent_manager):
        self.cfg = cfg
        self.user_agent_manager = user_agent_manager

    def build_search_url(self, query, num_images):
        params = OmegaConf.to_container(self.cfg.params, resolve=True)
        params["q"] = query
        params["count"] = str(
            min(num_images * self.cfg.count_multiplier, self.cfg.max_count)
        )
        return f"{self.cfg.url_base}?{urlencode(params)}"

    def fetch_image_html(self, url):
        self.user_agent_manager.update_headers()
        session = self.user_agent_manager.get_session()
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def extract_image_urls_from_soup(self, soup, num_images, extract_params):
        urls = []

        def add_url(candidate):
            if is_valid_image_url(candidate) and candidate not in urls:
                urls.append(candidate)
            return len(urls) >= num_images

        for link in soup.find_all(
            extract_params["link_tag"], class_=extract_params["link_class"]
        ):
            json_attr = link.get(extract_params["json_attr"])
            if json_attr:
                try:
                    data = json.loads(json_attr)
                    if extract_params["json_url_key"] in data and add_url(
                        data[extract_params["json_url_key"]]
                    ):
                        break
                except json.JSONDecodeError:
                    continue

        # Fallback - zwykłe <img>
        if len(urls) < num_images:
            for img in soup.find_all(extract_params["fallback_tag"]):
                for attr in extract_params["fallback_src_attrs"]:
                    src = img.get(attr)
                    if src and add_url(src):
                        break
                if len(urls) >= num_images:
                    break

        return urls[:num_images]

    def search_images(self, query, num_images):
        logger.info(f"Searching Images: {query}")
        try:
            url = self.build_search_url(query, num_images)
            soup = self.fetch_image_html(url)
            urls = self.extract_image_urls_from_soup(
                soup, num_images, self.cfg.extract_params
            )
            logger.info(f"Found {len(urls)} image URLs")
            return urls
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
