from bow import *
import yaml


with open("bow_averages.yaml") as file:
    yaml.safe_dump(averaged_data, file)