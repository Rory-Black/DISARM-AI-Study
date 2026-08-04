import json
import time
import requests
import os
from openai import OpenAI
from openai import APITimeoutError
from enum import Enum

class Mode(Enum):
    INVESTIGATE = 1 # prompt the model to catch first-hand techniques deployed by the authors of the article content
    RECOGNISE = 2 # prompt the model to catch meta-techniques reported in the article content 

TIMEOUT=600
OPENWEBUI_URL = "https://ai.datagaucho.com"
VLLM_URL = "http://localhost:8000"

local_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    timeout=TIMEOUT
)

gpt_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],   
    timeout=TIMEOUT
)

# SYSTEM PROMPTS

TA_SYS_FORMAT = """
{
Tactics: [{'name': "tactic name", 'description': "tactic description"}],
Article: "article text"
}
The agent MUST respond with the following JSON format: 
{
“Tactics”: [“List of tactic names”]
}
"""
T_SYS_FORMAT = """
{
Article: "article text"
Techniques: [{'external_id': "external_id"}, 'name': "technique name", 'description': "technique description"],
}
The agent MUST respond with the following JSON format: 
{
“Techniques”: [“List of external_id”]
}
"""

TA_SYS_INVEST = """""" + TA_SYS_FORMAT # TODO!!
TA_SYS_RECOG = """
The AI assistant has been designed to understand and categorize user input by the given Tactics. 
When processing user input, the assistant must predict the Tactics from one of the pre-defined options specified. 
It is essential to note that an article may have multiple Tactics associated. If the user input is not relevant to any Tactics, 
the assistant should print nothing, indicating that the input does not align with the available categories. 
The user input will be in the following format:
""" + TA_SYS_FORMAT

T_SYS_INVEST = """""" + T_SYS_FORMAT # TODO!!
T_SYS_RECOG = """
The AI assistant has been designed to understand and categorize user input by the given techniques. When processing user input, the assistant must predict the techniques from one of the pre-defined options specified. It is essential to note that an article may have multiple Techniques associated. If the user input is not relevant to any techniques, the assistant should print nothing, indicating that the input does not align with the available categories. 
The user input will be in the following format:
""" + T_SYS_FORMAT

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

# returns the external_id for a technique
def get_mitre_external_id(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None

class DISARM_LLM:
    def __init__(self,
                 article_content="",
                 model_name="",
                 local_model=True,
                 mode=Mode.INVESTIGATE,
                 check_sub_techniques=True
                 ):
        self.article_content = article_content
        self.MODEL_NAME = model_name
        self.local_model = local_model
        self.clf_mode = mode
        self.chq_sb_tchnqs = check_sub_techniques

        with open("DISARM.json", "r", encoding="utf-8") as f:
            self.disarm_json = json.load(f)
        pass

    def vllm_response(self, model_name, messages, response_format, seed=44):
        response = local_client.chat.completions.create(
            model=model_name,
            messages=messages,
            seed=seed,
            response_format=response_format,
            timeout=TIMEOUT
        )
        return response

    def chatGPT_response(self, messages, response_format, seed=44):
        response = gpt_client.chat.completions.create(
                model = self.MODEL_NAME,
                messages=messages,
                seed=seed,
                response_format=response_format,
                timeout=TIMEOUT
            )
        return response

    def prompt_llm_response(self, prompt, system_prompt=None, messages = None, response_format=None):
        start_time = time.time()
        print(f"{self.MODEL_NAME} thinking...")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        if self.local_model:
            response = self.vllm_response(messages=messages, response_format=response_format)
        else:
            response = self.chatGPT_response(messages=messages, response_format=response_format)

        full_response = ""

        # for non-streaming use
        full_response = response.choices[0].message.content
        print(full_response)

        print()
        print("\nDone in", round(time.time() - start_time, 2), "seconds")

        return full_response

    
    def prompt_valid_DISARM_response(self, prompt, system_prompt):
        # ensure response is JSON
        result_raw = self.prompt_llm_response(prompt, system_prompt)
        try:
            result_parsed = json.loads(result_raw)       
        except json.JSONDecodeError:
            print("Error: Model Failed to return valid JSON")
        # ensure response only contains valid classes
        if self.valid_classifications(result_parsed):
            return result_parsed
        else: raise Exception("Error: Model failed to return valid classifications")

    # helper method - returns true if json contains valid DISARM parent classes
    def valid_classifications(self, json_results):

        if not isinstance(json_results, dict):
            return False

        # Must contain exactly one of these keys
        valid_keys = ["Tactics", "Techniques"]
        present_keys = [k for k in valid_keys if k in json_results]

        if len(present_keys) != 1:
            return False

        key = present_keys[0]
        values = json_results[key]

        if not isinstance(values, list):
            return False

        # All returned items must be valid
        for i, v in enumerate(values):
            if v not in self.available_classes:
                # correct to parent technique if wrong
                if key == "Techniques":
                    parent_tech = v.split('_')[0]
                    if parent_tech in self.available_classes:
                        self.log_warning("Warning: Model returned invalid sub-technique, but parent technique is correct, ignoring...")
                        if parent_tech in values:
                            values.remove(v)
                        else:
                            values[i] = parent_tech
                        continue
                return False

        # final check
        if not all(v in self.available_classes for v in values):
            return False

        return True

    # retrieves tactic names and descriptions from JSON 
    # returns tactic data and accepted classification labels
    def get_tactics(self):
        tactics = []
        for obj in self.disarm_json["objects"]:
            if obj.get("type") == "x-mitre-tactic":
                tactics.append(obj)

        filtered_tactics = []
        accepted_labels = []

        # filter the tactic json data by name and description
        for tactic in tactics:
            filtered_tactics.append({
                "name": tactic.get("x_mitre_shortname"),
                "description": tactic.get("description")
            })
            accepted_labels.append(tactic.get("x_mitre_shortname"))

        return filtered_tactics, accepted_labels

    def get_all_techniques(self):
        techniques = []
        for obj in self.disarm_json["objects"]:
            if obj.get("type") == "attack-pattern":
                techniques.append(obj)
        return techniques

    # helper method for rationale extraction
    # returns the technique's description provided it's id
    def get_t_desc_from_id(self, external_ids:list, get_name=False):
        filtered_techniques = []
        for obj in self.disarm_json["objects"]:
            if obj.get("type") == "attack-pattern":
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        if ref.get("external_id") in external_ids:
                            filtered_techniques.append({
                                "external_id": (obj.get("name") if get_name else ref.get("external_id")),
                                "description": obj.get("description"),
                            })
        return filtered_techniques
    # prompts the llm to identify the DISARM tactics in the article
    def identify_tactics(self):
        
        filtered_tactics, available_classifications = self.get_tactics()

        self.available_classes = available_classifications
        ta_format = {
            "type": "object",
            "properties": {
                "Tactics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": self.available_classes
                    }
                }
            },
            "required": ["Tactics"]
        }

        # alter the system prompt depending on mode
        if self.clf_mode == Mode.INVESTIGATE:
            system_prompt = TA_SYS_INVEST
        else:
            system_prompt = TA_SYS_RECOG
        
        self.response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tactics_schema",
                "schema": ta_format
                }
            }

        ta_prompt = f"""
{{
"Article": {json.dumps(self.article_content)},
"Tactics": {json.dumps(filtered_tactics, indent=2)}
}}
"""
        
        print("Identifying Tactics...")
        ta_result_parsed = self.prompt_valid_DISARM_response(prompt=ta_prompt, system_prompt=system_prompt)
        ta_result_list = ta_result_parsed.get("Tactics", [])    
        print("Identified Tactics: " + str(ta_result_list))
        return ta_result_list

    # prompts the model to label the article with the provided techniques
    # leave techniques param empty to use select_all clf
    def identify_techniques(self, techniques=None, filter_ids=None):
        select_all = False
        if techniques is None:
            select_all = True
            techniques = self.get_all_techniques()

        filtered_techniques = []

        external_ids = []
        for technique in techniques:
            ex_id = get_mitre_external_id(technique)
            ex_id_token = ex_id.replace('.','d',1) #tokenised form to stop separate tokens at the '.'

            if filter_ids is None:
                filtered_techniques.append({
                    "external_id": ex_id_token,
                    "description": technique.get("description"),
                })
                external_ids.append(ex_id_token)
            elif ex_id in filter_ids:
                # filtering for use within reduced single clf in the ZeDPEB benchmark
                filtered_techniques.append({
                    "external_id": ex_id_token,
                    "description": technique.get("description"),
                })
                external_ids.append(ex_id_token)


        self.available_classes = external_ids

        t_format = {
            "type": "object",
            "properties": {
                "Techniques": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": external_ids
                    }
                }
            },
            "required": ["Techniques"]
        }
        
        # alter the system prompt depending on mode
        if self.clf_mode == Mode.INVESTIGATE:
            system_prompt = T_SYS_INVEST
        else:
            system_prompt = T_SYS_RECOG

        self.response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "techniques_schema",
                    "schema": t_format
                    }
                }
        # if it is a select_all setting, then swap the prompt structure to use prefix caching
        if select_all:
            t_prompt = f"""
{{
"Techniques": {json.dumps(filtered_techniques, indent=2)},
"Article": {json.dumps(self.article_content)}
}}
"""
        else:
            t_prompt = f"""
{{
"Article": {json.dumps(self.article_content)},
"Techniques": {json.dumps(filtered_techniques, indent=2)}
}}
"""
        try:
            t_result_parsed = self.prompt_valid_DISARM_response(prompt=t_prompt, system_prompt=system_prompt)
            t_result_list = t_result_parsed.get("Techniques", [])
            results = []
            for t in t_result_list:
                results.append(t.replace('d','.',1)) #return to untokenised form
        except APITimeoutError:
            print("Model Failed to return result")
            raise APITimeoutError
        print("Identified Techniques: " + str(results))
        return results
    
    # batchClf helper method
    # prompts the model to label the techniques belonging to a tactic 
    def identify_techniques_for_tactic(self, tactic):
        print("Testing for tactic: " + tactic)
        techniques = []
        for obj in self.disarm_json["objects"]:
            if obj.get("type") == "attack-pattern":
                for phase in obj.get("kill_chain_phases", []):
                    if phase.get("phase_name") == tactic:
                        techniques.append(obj)

        return self.identify_techniques(techniques)

    # prompts the model to return it's rationale behind each label
    def identify_rationales(self, external_ids):
        pass # TODO improvement

    # main methods

    # uses a batch_clf architecture to identify techniques, grouped by their associated tactics
    # fast setting uses a heirarchical approach to identify tactics first
    def batch_clf(self, fast=False):
        start_time = time.time()

        #TACTICS
        if fast:
            tactics = self.identify_tactics()
        else:
            f, tactics = self.get_tactics()
        #TECHNIQUES LOOP
        total_techniques = []
        for tactic in tactics:
            techniques = self.identify_techniques_for_tactic(tactic)
            total_techniques.extend(techniques)

        print("Total Identified Techniques: " + str(total_techniques))
        print("\nTotal execution time: " + str(round(time.time() - start_time, 2)) + " seconds")
        self.log_final_result(tactics,total_techniques, round(time.time() - start_time, 2))
        return tactics, total_techniques

    # uses select_all_clf architecture to identify all tecnhiques in a single prompts
    def select_all_clf(self):
        start_time = time.time() 

        total_techniques = self.identify_techniques()

        print("Total Identified Techniques: " + str(total_techniques))
        print("\nTotal execution time: " + str(round(time.time() - start_time, 2)) + " seconds")
        self.log_final_result(None,total_techniques, round(time.time() - start_time, 2))
        return total_techniques

    # one prompt for each tecnhique architecture
    def single_clf(self):
        start_time = time.time() 
        available_techniques = self.get_all_techniques()
        total_techniques = []
        for technique in available_techniques:
            total_techniques += self.identify_techniques(techniques=[technique])

        print("Total Identified Techniques: " + str(total_techniques))
        print("\nTotal execution time: " + str(round(time.time() - start_time, 2)) + " seconds")
        self.log_final_result(None,total_techniques, round(time.time() - start_time, 2))
        return total_techniques