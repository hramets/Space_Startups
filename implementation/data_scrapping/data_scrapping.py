import requests
import bs4
from bs4 import BeautifulSoup
import json
from typing import Any
from urllib.parse import urljoin
import logging


def parse_url(url: str) -> BeautifulSoup:
    """
    Method takes url, creates response with requests,
    parses to html with BeautifulSoup.
    Returns BeautifulSoup parser
    """
    response: requests.models.Response = requests.get(url)
    response.raise_for_status()

    parser: BeautifulSoup = BeautifulSoup(
        markup=response.text,
        features="html.parser")

    return parser


def extract_country_startups_page_info(
    page_url: str,
    storage: list[dict[str, Any]]
) -> None:
    """
    Function takes a certain country startups page url
    and parses to html. It extracts all the startups html blocks
    and then iterate through them. For each startup html block is
    used function 'extract_start_up_info', that extracts
    all the information about a certain startup.
    Extracted startup info is appended to the storage list.
    """
    try:
        country_startups_page_parser = parse_url(page_url)
    except requests.exceptions.RequestException as error:
        logging.warning(msg=f"Error fetching country URL {page_url}: {error}")
        return

    # All the startups html -> class "industries-inner"
    startups_info_html: bs4.element.ResultSet = (
        country_startups_page_parser.find_all(class_="industries-inner")
    )
    if not startups_info_html:
        logging.warning(msg=f"No startups found on {page_url}")
        return

    for startup_info_html in startups_info_html:
        startup_info: dict[str, str] = extract_startup_info(
            html=startup_info_html
        )
        storage.append(startup_info)


def extract_startup_info(html: bs4.element.ResultSet) -> dict[str, str]:
    """
    Function takes a startup html block extracting information about it.
    Returns startup info in dictionary.
    """
    startup_info: dict[str, Any] = {}
    try:
        # Extract startup name and idea.
        # Startup name appears on h2 tag in startup block.
        startup_name: str = html.h2.text if html.h2 else "Unknown"
        # Startup idea text appears in next sibling after
        # first appearance of class "their-mission_text".
        startup_idea_html: bs4.element.ResultSet = html.find(
            class_="their-mission_text"
        )
        startup_idea_text: str = startup_idea_html.next_sibling.text if (
            startup_idea_html and startup_idea_html.next_sibling
            ) else "Unknown"

        startup_info["Name"] = startup_name
        startup_info["Idea"] = startup_idea_text

        # All startup additional info parts are in classes "company_info".
        info_parts_html: bs4.element.ResultSet = html.find_all(
            class_="company-info"
        )

        # Extract additional info
        for info_html in info_parts_html:
            # Info parts text example: "Location:Australia"
            info_part: list[str] = (
                info_html.text.split(sep=":") if info_html else [
                    "Unknown", "Unknown"
                ]
            )

            if len(info_part) == 2:
                startup_info[info_part[0]] = info_part[1]
            else:
                logging.warning(
                    msg=f"Unexpected info part for {startup_info['Name']}"
                )

    except Exception as error:
        logging.warning(msg=f"Error extracting startup info: {error}")
        startup_info["Error"] = "Parsing failed"

    return startup_info


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        filename="C:\\Users\\artjo\\.vscode\\Space StartUps\\implementation\\data_scrapping\\logs.log",
        format="%(levelname)s - %(message)s\tLine: %(lineno)s",
        filemode="w"
    )

    url_base: str = "https://www.spacebandits.io"
    countries_page_url: str = urljoin(
        base=url_base,
        url="/startups-by-country"
    )
    all_startups_info_storage: list[dict[str, Any]] = []

    try:
        countries_page_parser = parse_url(url=countries_page_url)
    except requests.exceptions.RequestException as error:
        logging.critical(
            msg=f"Error fetching countries URL {countries_page_url}: {error}"
        )
        return

    countries: bs4.element.ResultSet = countries_page_parser.find_all(
        class_="w-dyn-item"
    )
    if not countries:
        logging.critical(msg=f"No countries found on {countries_page_url}")
        return

    country_startups_page_url: str = None
    country_startups_page_url_end: str = None
    for country in countries:
        # "a" tag with "href" contains the end of country startup url.
        if not (country.a or country.a["href"]):
            logging.warning(msg="A country page or url was not found")
            continue
        country_startups_page_url_end = country.a["href"].strip()
        country_startups_page_url = urljoin(
            base=url_base,
            url=country_startups_page_url_end
        )

        extract_country_startups_page_info(
            page_url=country_startups_page_url,
            storage=all_startups_info_storage
        )

    # Save data to json
    with open(
        "C:\\Users\\artjo\\.vscode\\Space StartUps\\assets\\data\\startups_data.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            obj=all_startups_info_storage,
            fp=file,
            indent=1,
            ensure_ascii=False
        )


if __name__ == "__main__":
    main()
