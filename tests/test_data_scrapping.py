import unittest
from implementation.data_scrapping.data_scrapping import (
    parse_url,
    extract_country_startups_page_info,
    extract_startup_info,
    extract_startup_industry,
    extract_startup_url
)
from bs4.element import ResultSet
from bs4 import BeautifulSoup
from typing import Callable


class TestDataScrappingFunctions(unittest.TestCase):

    def test_parse_url(self):
        func: Callable = parse_url
        test_url: str = (
            "https://www.goodreads.com/book/show/6752187-introduction-to-algorithms"
        )
        self.assertEqual(first=type(func(
            url=test_url
                )
            ), second=BeautifulSoup,
            msg="Wrong returned type. Error"
        )

    def test_extract_country_startups_page_info(self):
        func: Callable = extract_country_startups_page_info
        test_url: str = (
            "https://www.goodreads.com/book/show/6752187-introduction-to-algorithms"
        )
        with self.assertLogs(level="WARNING") as log:
            func(
                page_url=test_url,
                storage=[]
            )

        self.assertIn(
            member=f"No startups found on {test_url}",
            container=log.output[0],
            msg="Log not found"
        )

    def test_extract_startup_info(self):
        func: Callable = extract_startup_info
        with self.assertLogs(level="WARNING") as log:
            func(html=None)

        self.assertTrue(
            expr=any(
                "Error extracting startup info" in log for log in log.output
            ),
            msg="No logs occur while error with extracting startup info"
        )

    def test_extract_startup_industry(self):
        func: Callable = extract_startup_industry
        startup_url: str = (
            "https://www.spacebandits.io/startups/space-dynamix-marketing"
        )
        expected_type: type = str
        expected_value: str = "Information Research"

        self.assertEqual(
            first=type(func(startup_url=startup_url)),
            second=expected_type,
            msg=f"Extracted value type is not str: {
                type(func(startup_url=startup_url))
            }"
        )

        self.assertEqual(
            first=func(startup_url=startup_url),
            second=expected_value,
            msg=f"Wrong object is extracted: {func(startup_url=startup_url)}"
        )

    def test_extract_startup_url(self):
        func: Callable = extract_startup_url
        expected_url = (
            "https://www.spacebandits.io/startups/space-dynamix-marketing"
        )
        country_url: str = "https://www.spacebandits.io/countries/australia"
        country_startups_parser: BeautifulSoup = parse_url(url=country_url)

        startups_info_html: ResultSet = (
            country_startups_parser.find_all(class_="industries-inner")
        )
        startup_html_part: ResultSet = startups_info_html[0]

        url: str = func(startup_html=startup_html_part)

        with self.assertNoLogs(level="INFO"):
            func(startup_html=startup_html_part)

        self.assertEqual(
            first=url,
            second=expected_url,
            msg=f"Wrong url extracted: {url}"
        )


if __name__ == "__main__":
    unittest.main()
