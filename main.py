import coremltools as ct
from coremotiontools import convert
from tensoract.applications.vgg16 import VGG16


if __name__ == '__main__':
    model = VGG16(include_top=True)

    class_labels = ["stay", "walk", "jog", "skip", "stUp", "stDown"]
    classifier_config = ct.ClassifierConfig(class_labels=class_labels)
    mlmodel = convert(model, classifier_config=classifier_config)
    mlmodel.save("VGG16.mlmodel")
