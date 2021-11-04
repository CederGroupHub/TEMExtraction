import abc
import urllib.request


class BaseSoup(object):
    
    @property
    def base_url(self):
        raise NotImplementedError

    def create_full_url(self, partial: str) -> str:
        if not partial.startswith("http"):
            return self.base_url + partial
        else:
            return partial

    def parse(self, html_str, **kwargs):
        results = self._parse(html_str, **kwargs)
        return results

    @staticmethod
    @abc.abstractmethod
    def _parse(html_str):
        raise NotImplementedError
