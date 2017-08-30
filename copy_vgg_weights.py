import keras.applications.resnet50


print("Loading weights during Docker build: ")
keras.applications.resnet50.ResNet50(include_top=False)