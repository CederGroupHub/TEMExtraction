from classifier.builders import factory

def classify(data_path: str, classifier: str, output_dir: str, GPU: bool):
    evaluator = factory.create(classifier, data_path, output_dir)
    evaluator.infer(GPU)
