class BuilderFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, class_name):
        self._builders[key] = class_name

    def create(self, key, data_path, output_dir):
        builder = self._builders.get(key)
        return builder(data_path, output_dir)
