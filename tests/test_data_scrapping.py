import unittest
from implementation.data_scrapping.data_scrapping import *
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
            any("Error extracting startup info" in log for log in log.output)
        )
        
        
if __name__ == "__main__":
    unittest.main()