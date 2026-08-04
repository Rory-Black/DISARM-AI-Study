import json
import time
import requests
from openai import OpenAI
from openai import APITimeoutError

TIMEOUT=600
OPENWEBUI_URL = "https://ai.datagaucho.com"
VLLM_URL = "http://localhost:8000"

TA_SYSTEM_PROMPT_FORMAT = """
The AI assistant has been designed to understand and categorize user input by the given Tactics. 
When processing user input, the assistant must predict the Tactics from one of the pre-defined options specified. 
It is essential to note that an article may have multiple Tactics associated. If the user input is not relevant to any Tactics, 
the assistant should print nothing, indicating that the input does not align with the available categories. 
The user input will be in the following format:
{
Tactics: [{'name': "tactic name", 'description': "tactic description"}],
Article: "article text"
}
The agent MUST respond with the following JSON format: 
{
“Tactics”: [“List of tactic names”]
}
"""
T_SYSTEM_PROMPT_FORMAT = """
The AI assistant has been designed to understand and categorize user input by the given techniques. When processing user input, the assistant must predict the techniques from one of the pre-defined options specified. It is essential to note that an article may have multiple Techniques associated. If the user input is not relevant to any techniques, the assistant should print nothing, indicating that the input does not align with the available categories. 
The user input will be in the following format:
{
Article: "article text"
Techniques: [{'external_id': "external_id"}, 'name': "technique name", 'description': "technique description"],
}
The agent MUST respond with the following JSON format: 
{
“Techniques”: [“List of external_id”]
}
"""
SYS_DSTNKT = """
CRITICAL DISTINCTION:

If the article describes, reports on, or paraphrases disinformation used by others, this must NOT be classified as disinformation.
If the article itself uses manipulative, misleading, or deceptive techniques, then those techniques should be classified.

The assistant must therefore distinguish between:

- Descriptive content (reporting on disinformation) -> no classification
- Performative content (using disinformation techniques) -> classify techniques

If no techniques are directly used by the author, the assistant must output nothing.

An article may contain multiple techniques if and only if they are present in the author’s own communication style or argumentation.
"""

T_RAT_SYSTEM_PROMPT = """
The AI assistant has been designed to understand and categorize user input by the given techniques. 
When processing user input, for each technique given by the user, the assistant must select quotes from the article that represent the rationale behind each technique classification. 
It is essential to note that an article may have multiple Techniques associated which may share rationales with others.
The user input will be in the following format:
{
    Article: "article text",
    Techniques: [{'external_id': "external_id"}, 'name': "technique name", 'description': "technique description"]
}
The agent MUST respond with the following JSON format: 
{
  "results": [
    {
      "technique": "external_id",
      "quotes": [
        "quote 1",
        "quote 2"
      ]
    }
  ]
}
The agent MUST 
- return one object per technique
- include at least one quote per technique
- ensure quotes are direct substrings of the article
The agent should be greedy in it's quote selection: quote anything that would support the classification.
"""
RAT_FORMAT_SYS_PROMPT = """
The AI assistant has been designed to format user input. 
When processing user input, for each technique given by the user, the assistant must format the user input into the the provided JSON format.
The agent MUST respond with the following JSON format: 
{
  "results": [
    {
      "technique": "external_id",
      "quotes": [
        "quote 1",
        "quote 2"
      ]
    }
  ]
}
The agent MUST 
- return one object per technique
- include at least one quote per technique if the previous model response contains adequate information, otherwise leave as an empty string.
"""