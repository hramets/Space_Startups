import requests
from bs4.element import ResultSet, Tag
from bs4 import BeautifulSoup
import json
from typing import Any
from urllib.parse import urljoin
import logging


error_logger: logging.Logger = logging.getLogger(
    name="data_scrapping_errors"
)
error_logger.setLevel(logging.ERROR)
error_handler: logging.FileHandler = logging.FileHandler(
    filename="data_scraping\error_logger.log", mode="w"
)
error_formatter = logging.Formatter(
    fmt="%(name)s - Line: %(lineno)s\n%(message)s"
)
error_handler.setFormatter(fmt=error_formatter)
error_logger.addHandler(hdlr=error_handler)

info_logger: logging.Logger = logging.getLogger(
    name="data_scrapping_info"
)
info_logger.setLevel(logging.INFO)
info_handler: logging.FileHandler = logging.FileHandler(
    filename="data_scraping\debug_logger.log", mode="w"
)
info_formatter = logging.Formatter(
    fmt="%(name)s - %(message)s"
)
info_handler.setFormatter(fmt=info_formatter)
info_logger.addHandler(hdlr=info_handler)


def parse_url(url: str) -> BeautifulSoup:
    """
    Function takes url, creates response with requests and
    parses to html with BeautifulSoup.
    Returns BeautifulSoup parser.
    """
    response: requests.models.Response = requests.get(url)
    response.raise_for_status()

    parser: BeautifulSoup = BeautifulSoup(
        markup=response.text,
        features="html.parser"
    )

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
    all the information about a startup.
    Extracted startup info is appended to the storage list.
    Returns the storage list.
    """
    try:
        country_startups_page_parser: BeautifulSoup = parse_url(page_url)
    except requests.exceptions.RequestException as error:
        error_logger.error(
            msg=f"Error fetching country URL {page_url}: {error}"
        )
        return

    # All the startups html -> class "industries-inner"
    startups_info_html: ResultSet = (
        country_startups_page_parser.find_all(class_="industries-inner")
    )
    if not startups_info_html:
        info_logger.warning(msg=f"No startups found on {page_url}")
        return

    for startup_info_html in startups_info_html:
        startup_info: dict[str, str] = extract_startup_info(
            html=startup_info_html
        )
        storage.append(startup_info)


def extract_startup_info(html: ResultSet) -> dict[str, str]:
    """
    Function takes a startup html block to extract it's information.
    Returns startup info as dictionary.
    """
    startup_info: dict[str, Any] = {}
    try:
        # Extract startup name and idea.
        # Startup name appears on h2 tag in startup block.
        startup_name: str = html.h2.text if html.h2 else "Unknown"
        # Startup idea text appears in next sibling after
        # first appearance of class "their-mission_text".
        startup_idea_html: ResultSet = html.find(
            class_="their-mission_text"
        )
        startup_idea_text: str = startup_idea_html.next_sibling.text if (
            startup_idea_html and startup_idea_html.next_sibling
            ) else "Unknown"
        startup_info["Name"] = startup_name
        startup_info["Idea"] = startup_idea_text

        # Extract startup industry.
        # Startup industry is on the startup page.
        startup_url: str = extract_startup_url(startup_html=html)
        if not startup_url:
            info_logger.warning(
                msg=f"startup url was not found for {startup_name}"
            )

        startup_info["Industry"] = (
            extract_startup_industry(startup_url=startup_url)
        )

        # All startup additional info parts are in classes "company_info".
        info_parts_html: ResultSet = html.find_all(
            class_="company-info"
        )

        # Extract additional info
        for info_html in info_parts_html:
            # Info parts text example: "Location:Australia"
            info_part: list[str] = (
                info_html.text.split(sep=":") if info_html else [
                    "Unknown",
                    "Unknown"
                ]
            )

            if len(info_part) == 2:
                startup_info[info_part[0]] = info_part[1]
            else:
                error_logger.error(
                    msg=f"Unexpected info part for {startup_info['Name']}"
                )

    except Exception as error:
        error_logger.error(msg=f"Error extracting startup info: {error}")
        startup_info["Error"] = "Parsing failed"

    return startup_info


def extract_startup_industry(startup_url: str) -> str:
    """
    Function takes a startup page url.
    Extracts from the page startup's industry and returns it.
    """
    try:
        startup_page_parser: BeautifulSoup = parse_url(startup_url)
    except requests.exceptions.RequestException as error:
        error_logger.error(
            msg=f"Error fetching startup URL {startup_url}: {error}"
        )
        return

    startup_industry_html: ResultSet = (
        startup_page_parser.find(class_="pill blue")
    )
    if not startup_industry_html:
        info_logger.warning(
            msg=f"Startup industry was not found on {startup_url}"
        )
        return "Unknown"

    industry: str = startup_industry_html.text

    return industry


def extract_startup_url(
    startup_html: Tag,
    url_base: str = "https://www.spacebandits.io"
) -> str:
    """
    Function extracts startup url ending from a startup html part and
    concatenates this with url base.
    Returns full startup url.
    """
    url_end: str = startup_html.a["href"]

    if not url_end:
        return

    return url_base + url_end


def main() -> None:

    url_base: str = "https://www.spacebandits.io"
    countries_page_url: str = urljoin(
        base=url_base,
        url="/startups-by-country"
    )
    all_startups_info_storage: list[dict[str, Any]] = []

    try:
        countries_page_parser = parse_url(url=countries_page_url)
    except requests.exceptions.RequestException as error:
        error_logger.critical(
            msg=f"Error fetching countries URL {countries_page_url}: {error}"
        )
        return

    countries: ResultSet = countries_page_parser.find_all(
        class_="w-dyn-item"
    )
    if not countries:
        error_logger.critical(
            msg=f"No countries found on {countries_page_url}."
        )
        return

    for country in countries:
        # "a" tag with "href" contains the end of country startup url.
        if not (country.a or country.a["href"]):
            info_logger.warning(
                msg=f"A country {country} page or url was not found."
            )
            continue
        country_startups_page_url_end: str = country.a["href"].strip()
        country_startups_page_url: str = urljoin(
            base=url_base,
            url=country_startups_page_url_end
        )

        extract_country_startups_page_info(
            page_url=country_startups_page_url,
            storage=all_startups_info_storage
        )

    # Save data to json.
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
