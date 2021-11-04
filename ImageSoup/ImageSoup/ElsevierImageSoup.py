import re
import json
import urllib.request
from bs4 import BeautifulSoup

from ImageSoup.ImageSoup.BaseSoup import BaseSoup

# curl -o x -X GET --header 'Accept: image/jpeg' 'https://api.elsevier.com/content/object/eid/1-s2.0-S0926337317307567-fx1.sml?apiKey=7f59af901d2d86f78a1fd60c1bf9426a'


class ElsevierSoup(BaseSoup):

    def _ref_to_url(self, paper):
        ref_to_url = {}
        objects = paper.find_all('object', attrs={'category': re.compile('thumbnail')})
        for obj in objects:
            ref = obj.get('ref')
            url = obj.get_text()
            ref_to_url[ref] = url
        return ref_to_url

    def _extract_images(self, images, ref_to_url):
        image_meta = {'figures':[]}
        for image in images:
            # URL
            ref = image.find('ce:link').get('locator')
            url = ref_to_url[ref]
            id = re.search(r'^https://api.elsevier.com/content/object/eid/(.*)\?httpAccept.*$', url).group(1)
            img_url = 'https://ars.els-cdn.com/content/image/' + id

            # Caption
            try:
                caption = image.find('ce:caption').get_text().strip()
            except:
                continue

            # Title
            title = image.find('ce:label').get_text()

            image_meta['figures'].append({'Image_URL': img_url, 'Caption': caption, 'Title': title})
        return image_meta

    def _parse(self, html_string, **kwargs) -> dict:
        paper = BeautifulSoup(html_string, 'lxml-xml')
        ref_to_url = self._ref_to_url(paper)

        images = paper.find_all('ce:figure')
        return self._extract_images(images, ref_to_url)
