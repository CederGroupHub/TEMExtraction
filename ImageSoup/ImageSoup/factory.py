from ImageSoup.ImageSoup import ElsevierSoup, NatureSoup, RSCSoup, SpringerSoup

class Factory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, class_name):
        self._builders[key] = class_name

    def create(self, key):
        builder = self._builders.get(key)
        print(key, builder)
        return builder()

factory = Factory()
factory.register_builder('elsevier', ElsevierSoup)
factory.register_builder('nature', NatureSoup)
factory.register_builder('rsc', RSCSoup)
factory.register_builder('springer', SpringerSoup)

key_to_publisher = {
    'elsevier': 'Elsevier',
    'nature': 'Nature Publishing Group',
    'rsc': 'The Royal Society of Chemistry',
    'springer': 'Springer'
}
