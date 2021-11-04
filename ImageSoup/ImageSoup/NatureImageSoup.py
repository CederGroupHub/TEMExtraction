import re
import json
import urllib.request
from bs4 import BeautifulSoup

from ImageSoup.ImageSoup.BaseSoup import BaseSoup


class NatureSoup(BaseSoup):

    journal_to_id = {
        "srep": 41598,
        "nature": 41586,
        "ncomms": 41467,
        "nchem": 41557,
        "nmat": 41563,
        "nnano": 41565,
        "nphys": 41567,
    }

    def get_id(self, sub_journal):
        if re.search(r'^s\d.*$', sub_journal):
            return re.search(r'^s(\d+)-.*$', sub_journal).group(1)
        else:
            return self.journal_to_id[re.search(r'^(\D+)\d+$', sub_journal).group(1)]

    def build_url(self, doi, fig_num, year):
        sub_journal = re.search(r'^10.1038/(.*)$', doi).group(1)

        if re.search(r'^s\d.*$', sub_journal):
            base = "https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2F{}/MediaObjects/{}_{}_{}_Fig{}_HTML.jpg?as:webp".format(
                sub_journal,
                self.get_id(sub_journal),
                '2' + re.search(r'^s\d+-(\d+)-.*$', sub_journal).group(1),
                re.search(r'^s\d+-\d+-0*(\d+)-.*$', sub_journal).group(1),
                fig_num,
            )
        else:
            base = "https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2F{}/MediaObjects/{}_{}_Article_BF{}_Fig{}_HTML.jpg?as:webp".format(
                sub_journal,
                self.get_id(sub_journal),
                year,
                sub_journal,
                fig_num,
            )
        return base

    def _extract_images(self, images, doi, year):
        image_meta = {'figures':[]}
        for image in images:
            # Title
            title = image.find('figcaption').get_text()
            if title.strip().startswith("Table"):
                continue

            # Caption
            try:
                caption = image.find('p').get_text()
            except:
                caption = title

            # URL
            fig_num = re.search(r'\d+', title).group()
            img_url = self.build_url(doi, fig_num, year)

            image_meta['figures'].append({'Image_URL': img_url, 'Caption': caption, 'Title': title})
        return image_meta

    def _parse(self, html_string, **kwargs) -> dict:
        paper = BeautifulSoup(html_string, 'html.parser')
        images = paper.find_all('figure')
        doi = kwargs.get('doi', None)
        year = kwargs.get('year', None)
        return self._extract_images(images, doi, year)
