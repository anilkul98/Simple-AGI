import os
import openai
import yaml
from yaml.loader import SafeLoader
from model import AGIModel

with open('credentials.yaml') as f:
    credentials = yaml.load(f, Loader=SafeLoader)
print(credentials)

openai.organization = credentials["organization_id"]
openai.api_key = credentials["api_key"]

agi = AGIModel(["1 - Object Detection", "2 - Image Classification"])
