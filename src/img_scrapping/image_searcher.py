from urllib.parse import urlencode

from bs4 import BeautifulSoup
from omegaconf import OmegaConf

from src.img_scrapping.image_extraction import extract_image_urls_from_soup
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

    def fetch_html(self, url):
        self.user_agent_manager.rotate_user_agent()
        session = self.user_agent_manager.get_session()
        response = session.get(url, timeout=self.cfg.timeout)
        response.raise_for_status()
        return BeautifulSoup(response.text, self.cfg.parser)

    def search_images(self, query, num_images):
        logger.info(f"Searching Images: {query}")
        url = self.build_search_url(query, num_images)
        soup = self.fetch_html(url)
        urls = extract_image_urls_from_soup(soup, num_images, self.cfg.extract_params)
        logger.info(f"Found {len(urls)} image URLs")
        return urls

    def manage_searching_images(self, query, num_images=10):
        all_urls = []

        logger.info("--- Trying Searching ---")
        urls = self.search_images(query, num_images)

        all_urls += urls
        all_urls = list(set(all_urls))

        logger.info(f"Added {len(urls)} new URLs")
        logger.info(f"Total found {len(all_urls)} unique URLs")
        return all_urls
