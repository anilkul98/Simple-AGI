from model import AGIModel
from problem_enum import ProblemType
from llm import LLM

problem_types = [pt.value for pt in ProblemType]
llm = LLM()
question = input("Enter question:")
answer = llm.get_answer(question)

if answer == "-1":
    print("Unrelated question")
else:
    print(f"[INFO] : llm answer => {answer}")
    probs_arr = answer.split("and")
    detected_problems = []
    for prob in probs_arr:
        if prob in problem_types:
            detected_problems.append(prob)
    if len(detected_problems) != 0:
        agi = AGIModel(detected_problems)
        results_dct = agi.get_results("sample_data/img-classification/hagnose-sneak.jpeg")
        print(results_dct)
    else:
        print(f"[ERROR] => answer : {answer}")
