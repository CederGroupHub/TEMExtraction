import re
import json
import urllib.request
from bs4 import BeautifulSoup

from ImageSoup.ImageSoup.BaseSoup import BaseSoup


class RSCSoup(BaseSoup):

    base_url = "https://pubs.rsc.org"

    def _extract_images(self, image_tables):
        image_meta = {'figures':[]}
        for image_table in image_tables:
            # URL
            partial = image_table.find('img').get('src')
            img_url = self.create_full_url(partial)

            # Caption
            try:
                caption = image_table.find('span', attrs={'class': 'graphic_title'}).get_text()
            except:
                continue
            
            # Title
            title = image_table.find('td', attrs={'class': 'image_title'}).find('b').get_text().strip()

            image_meta['figures'].append({'Image_URL': img_url, 'Caption': caption, 'Title': title})
        return image_meta

    def _parse(self, html_string, **kwargs) -> dict:
        paper = BeautifulSoup(html_string, 'html.parser')
        divs = paper.find_all('div')
        image_tables = paper.find_all('div', attrs={'class': 'image_table'})
        return self._extract_images(image_tables)
