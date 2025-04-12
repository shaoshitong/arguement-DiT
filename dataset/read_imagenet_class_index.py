import json

with open('imagenet.json') as json_file:
    labels_dict = json.load(json_file)