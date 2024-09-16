import openai
from langsmith.wrappers import wrap_openai

from dotenv import load_dotenv

from prompts import prompt1, prompt2, negative_control

# Load environment variables from .env file
load_dotenv()

openai_client = wrap_openai(openai.Client())

def answer_stem_analogy_question(inputs: dict) -> dict:
    """
    Parameters:
    inputs (dict): A dictionary with a single key 'question', representing the user's question as a string.

    Returns:
    dict: A dictionary with a single key 'output', containing the generated answer as a string.
    """

    # System prompt
    system_msg = negative_control
    

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": inputs["question"]},
    ]

    # Call OpenAI
    response = openai_client.chat.completions.create(
        messages=messages, model="gpt-3.5-turbo"
    )

    # Response in output dict
    return {"answer": response.dict()["choices"][0]["message"]["content"]}
  
  
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Evaluators
qa_evalulator = [LangChainStringEvaluator("cot_qa")]
dataset_name = "Analogies"

experiment_results = evaluate(
    answer_stem_analogy_question,
    data=dataset_name,
    evaluators=qa_evalulator,
    experiment_prefix="test-stem-analogies-qa-oai",
)