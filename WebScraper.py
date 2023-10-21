from bs4 import BeautifulSoup
import requests
import lxml


class WebScraper:
    def __init__(self, url: str):
        self.url = url
        self.soup = self.get_page()

    def get_page(self) -> BeautifulSoup:
        response = requests.get(self.url)
        if not response.ok:
            print("Server responded: ", response.status_code)
            return None
        else:
            soup = BeautifulSoup(response.text, "lxml")
            return soup

    def get_detailed_data(self, tag: str, class_: str = None) -> list:
        data = []
        try:
            for item in self.soup.find_all(tag, class_):
                data.append(item.text)
        except None:
            print("No data found")
        return data
