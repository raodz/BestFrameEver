from src.img_scrapping.image_downloader import ImageDownloader
from src.img_scrapping.image_searcher import ImageSearcher
from src.img_scrapping.user_agent_manager import UserAgentManager


class ImageScraper:
    """Main image scraper class that orchestrates the scraping process"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.user_agent_manager = UserAgentManager(cfg.user_agent_manager)
        self.searcher = ImageSearcher(cfg.image_searcher, self.user_agent_manager)
        self.downloader = ImageDownloader(
            cfg.image_downloader, self.user_agent_manager, self.searcher
        )

    def download_images_for_actor(self, actor_name, num_images, output_dir="data"):
        """Download images for actor to specified directory"""
        return self.downloader.download_images_for_actor(
            actor_name, num_images, output_dir
        )
