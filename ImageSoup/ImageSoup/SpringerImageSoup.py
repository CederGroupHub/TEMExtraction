import re
import json
import urllib.request
from bs4 import BeautifulSoup

from ImageSoup.ImageSoup.BaseSoup import BaseSoup


class SpringerSoup(BaseSoup):

    def _extract_images(self, images):
        image_meta = {'figures':[]}
        for image in images:
            # Title
            title = image.find('span', attrs={'class': 'CaptionNumber'}).get_text()

            # Caption
            caption = image.find('p', attrs={'class': 'SimplePara'}).get_text()

            # URL
            img_url = image.find('a').get('href')

            image_meta['figures'].append({'Image_URL': img_url, 'Caption': caption, 'Title': title})
        return image_meta

    def _parse(self, html_string, **kwargs) -> dict:
        paper = BeautifulSoup(html_string, 'html.parser')
        images = paper.find_all('figure')
        return self._extract_images(images)
