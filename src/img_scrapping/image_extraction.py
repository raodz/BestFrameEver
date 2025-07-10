import json

from bs4 import BeautifulSoup

from src.img_scrapping.image_validation import is_valid_image_url


def extract_from_json_links(
    soup: BeautifulSoup, ep: dict, urls: list[str], num_images: int
):
    """
    Extracts image URLs from <a> tags with embedded JSON attribute.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.
        ep (dict): Extraction parameters with keys:
            - "link_tag": tag to search (e.g., "a")
            - "link_class": class name to filter tags
            - "json_attr": attribute containing JSON (e.g., "m")
            - "json_url_key": key inside the JSON with image URL (e.g., "murl")
        urls (list[str]): List to append valid image URLs to.
        num_images (int): Max number of URLs to collect.
    """
    for link in soup.find_all(ep["link_tag"], class_=ep["link_class"]):
        raw_json = link.get(ep["json_attr"])
        if not raw_json:
            continue

        try:
            data = json.loads(raw_json)
            candidate = data.get(ep["json_url_key"])
        except json.JSONDecodeError:
            continue

        if candidate and is_valid_image_url(candidate) and candidate not in urls:
            urls.append(candidate)
            if len(urls) >= num_images:
                break


def extract_from_fallback_imgs(
    soup: BeautifulSoup, ep: dict, urls: list[str], num_images: int
):
    """
    Extracts image URLs from <img> tags using fallback attributes.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.
        ep (dict): Extraction parameters with keys:
            - "fallback_tag": tag to search (typically "img")
            - "fallback_src_attrs": list of attribute names to check (e.g., ["src", "data-src"])
        urls (list[str]): List to append valid image URLs to.
        num_images (int): Max number of URLs to collect.
    """
    for img in soup.find_all(ep["fallback_tag"]):
        for attr in ep["fallback_src_attrs"]:
            candidate = img.get(attr)
            if candidate and is_valid_image_url(candidate) and candidate not in urls:
                urls.append(candidate)
                break
        if len(urls) >= num_images:
            break


def extract_image_urls_from_soup(
    soup: BeautifulSoup, num_images: int, extract_params: dict
) -> list[str]:
    urls = []
    extract_from_json_links(soup, extract_params, urls, num_images)

    if len(urls) < num_images:
        extract_from_fallback_imgs(soup, extract_params, urls, num_images)

    return urls
