import os
import openai
import re

# Choosing a model, also available at https://platform.openai.com/docs/models
# They also differ in price and speed: https://openai.com/pricing#language-models
#openai.Model.list()
#https://platform.openai.com/docs/api-reference/completions/create?lang=python
openai.api_key = os.getenv("OPENAI_API_KEY")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")

def get_absa(
    text,
    temperature = 0, # [0, 2]: Makes the output more random
    max_tokens = 1024, # max = 2048
    frequency_penalty = 1, #[-2, 2]: The higher value decreases the model's likelihood to repeat the same line verbatim.
    presence_penalty = -1, #[-2, 2]: Increases the model's likelihood to talk about new topics
    model = FINE_TUNED_MODEL, #"ada", "babbage", "curie", "davinci", "text-davinci-002",
):
    prompt = f"{text}\n\n###\n\n"

    return openai.Completion.create(
        model = model,
        prompt = prompt,
        temperature = temperature,
        max_tokens = max_tokens,
        frequency_penalty = frequency_penalty,
        presence_penalty = presence_penalty,
        stop = [" END"],
    )

# temperature [0, 2]: Higher values like 0.8 will make the output more random, 
# while lower values like 0.2 will make it more focused and deterministic.