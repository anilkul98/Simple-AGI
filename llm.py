import os
import openai
import yaml
from yaml.loader import SafeLoader
from constants import PROMPT

with open('credentials.yaml') as f:
    credentials = yaml.load(f, Loader=SafeLoader)

class LLM():
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.prompt = PROMPT
        self.model_name = model_name
        self.login()
        
    def login(self):
        openai.organization = credentials["organization_id"]
        openai.api_key = credentials["api_key"]
    
    def get_answer(self, question):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": self.prompt + question}
            ]
        )
        answer = response["choices"][0]["message"]["content"]
        return answer