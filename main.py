import os
import openai
import yaml
from yaml.loader import SafeLoader
from model import AGIModel
from problem_enum import ProblemType

with open('credentials.yaml') as f:
    credentials = yaml.load(f, Loader=SafeLoader)

openai.organization = credentials["organization_id"]
openai.api_key = credentials["api_key"]

# agi = AGIModel(["1 - Object Detection"])
agi = AGIModel(["1 - Object Detection", "2 - Image Classification"])
results_dct = agi.get_results("sample_data/img-classification/hagnose-sneak.jpeg")
print(results_dct)
# res_labels = results_dct["labels"].cpu().detach().numpy()
# res_scores = results_dct["scores"].cpu().detach().numpy()
# print(tuple(zip([COCO_INSTANCE_CATEGORY_NAMES[ind] for ind in res_labels], res_scores)))

# problem_types = [pt.value for pt in ProblemType]

# question = input("Enter question:")

# prompt = f"""Could you define the computer vision tasks between following option depends on the question asked. There might be multiple tasks as an answer. Please answer only the number of the options. If the question is not related to topic just answer with "-1".
#             1 - Object Detection
#             2 - Image Classification

#             Question: “{question}”
#             """

# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "user", "content": prompt}
#     ]
# )
# answer = response["choices"][0]["message"]["content"]

# if answer == "-1":
#     print("Unrelated question")
# else:
#     probs_arr = answer.split("and")
#     detected_problems = []
#     for prob in probs_arr:
#         if prob in problem_types:
#             detected_problems.append(prob)
#     if len(detected_problems) != 0:
#         agi = AGIModel(detected_problems)
#     else:
#         print(f"[ERROR] => answer : {answer}")

