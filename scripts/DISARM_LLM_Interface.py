import json
import time
import requests
import os
from openai import OpenAI
from openai import APITimeoutError
from enum import Enum
from pathlib import Path

from DISARM_DATA_MASTER import get_mitre_external_id

class Mode(Enum):
    INVESTIGATE = 1 # prompt the model to catch first-hand techniques deployed by the authors of the article content
    RECOGNISE = 2 # prompt the model to catch meta-techniques reported in the article content 

TIMEOUT=600
OPENWEBUI_URL = "https://ai.datagaucho.com"
VLLM_URL = "http://localhost:8000"

JSON_DATA = Path(__file__).parent.parent / ".data" / "DISARM.json"


local_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    timeout=TIMEOUT
)

api_key = os.environ.get("OPENAI_API_KEY")

print("API key exists:", api_key is not None)
print("API key length:", len(api_key) if api_key else 0)

gpt_client = OpenAI(
    api_key=api_key,
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

TA_SYS_INVEST = """
# DISARM Disinformation tactic Classification

The AI assistant identifies DISARM disinformation tactics used by the **author or publisher of the analysed media content**.

When processing an article, select applicable tactics from the predefined options.

## Core Rule

Only classify a tactic when the **author or publisher themselves performs, facilitates, or directly contributes to the behaviour represented by that tactic**.

Do **not** classify a tactic merely because the article:

* Describes another actor using the tactic.
* Quotes or reports another actor's disinformation.
* Analyses or discusses a disinformation campaign.
* Contains information that was itself produced using a DISARM tactic by another actor.

The classification target is:

> **What DISARM tactics did the author/publisher use to create, manipulate, amplify, or distribute disinformation?**

Not:

> **What DISARM tactics are mentioned or attributed to actors in the article?**

## Attribution

For each candidate tactic, determine:

* **Who** performed the behaviour?
* **What** did they do?
* **Does the behaviour correspond to the tactic?**
* **Is there sufficient evidence to attribute it to the author/publisher?**

For example, if an article reports that Actor X fabricated information, do not classify the fabrication tactic unless the author/publisher also fabricated, promoted, or otherwise directly used that fabricated information in a way that constitutes the tactic.

Quoting a false claim does not automatically constitute use of the tactic. However, if the author presents, manipulates, reframes, amplifies, or distributes the claim in a manner that itself constitutes a DISARM tactic, it may be classified.

## Classification Rules

* Multiple tactics may be selected when the author/publisher uses multiple tactics.
* Do not infer tactic usage solely from the presence of disinformation or a tactic within the article.
* If attribution is ambiguous or insufficiently supported, **do not classify the tactic**.
* Prefer false negatives over incorrectly attributing a tactic to the author/publisher.
* If no applicable tactics are used by the author/publisher, return no tactics.

### Final Decision Rule

For every candidate tactic, ask:

> **Did the author/publisher perform this behaviour, or did they merely report that someone else performed it?**

**Author/publisher performed it → classify.**
**Another actor performed it → do not classify.**
**Author merely reported, quoted, or analysed it → do not classify.**
**Attribution is uncertain → do not classify.**

""" + TA_SYS_FORMAT # TODO!!
TA_SYS_RECOG = """
The AI assistant has been designed to understand and categorize user input by the given Tactics. 
When processing user input, the assistant must predict the Tactics from one of the pre-defined options specified. 
It is essential to note that an article may have multiple Tactics associated. If the user input is not relevant to any Tactics, 
the assistant should print nothing, indicating that the input does not align with the available categories. 
The user input will be in the following format:
""" + TA_SYS_FORMAT

T_SYS_INVEST = """
# DISARM Disinformation Technique Classification

The AI assistant identifies DISARM disinformation techniques used by the **author or publisher of the analysed media content**.

When processing an article, select applicable techniques from the predefined options.

## Core Rule

Only classify a technique when the **author or publisher themselves performs, facilitates, or directly contributes to the behaviour represented by that technique**.

Do **not** classify a technique merely because the article:

* Describes another actor using the technique.
* Quotes or reports another actor's disinformation.
* Analyses or discusses a disinformation campaign.
* Contains information that was itself produced using a DISARM technique by another actor.

The classification target is:

> **What DISARM techniques did the author/publisher use to create, manipulate, amplify, or distribute disinformation?**

Not:

> **What DISARM techniques are mentioned or attributed to actors in the article?**

## Attribution

For each candidate technique, determine:

* **Who** performed the behaviour?
* **What** did they do?
* **Does the behaviour correspond to the technique?**
* **Is there sufficient evidence to attribute it to the author/publisher?**

For example, if an article reports that Actor X fabricated information, do not classify the fabrication technique unless the author/publisher also fabricated, promoted, or otherwise directly used that fabricated information in a way that constitutes the technique.

Quoting a false claim does not automatically constitute use of the technique. However, if the author presents, manipulates, reframes, amplifies, or distributes the claim in a manner that itself constitutes a DISARM technique, it may be classified.

## Classification Rules

* Multiple techniques may be selected when the author/publisher uses multiple techniques.
* Do not infer technique usage solely from the presence of disinformation or a technique within the article.
* If attribution is ambiguous or insufficiently supported, **do not classify the technique**.
* Prefer false negatives over incorrectly attributing a technique to the author/publisher.
* If no applicable techniques are used by the author/publisher, return no techniques.

### Final Decision Rule

For every candidate technique, ask:

> **Did the author/publisher perform this behaviour, or did they merely report that someone else performed it?**

**Author/publisher performed it → classify.**
**Another actor performed it → do not classify.**
**Author merely reported, quoted, or analysed it → do not classify.**
**Attribution is uncertain → do not classify.**
 
""" + T_SYS_FORMAT # TODO!!
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

        with open(JSON_DATA, "r", encoding="utf-8") as f:
            self.disarm_json = json.load(f)
        pass

    def vllm_response(self, messages, response_format, seed=44):
        response = local_client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=messages,
            seed=seed,
            response_format=response_format,
            timeout=TIMEOUT
        )
        return response

    def chatGPT_response(self, input, system_prompt, tools, response_format):
        response = gpt_client.responses.create(
                model = self.MODEL_NAME,
                input=input,
                instructions=system_prompt,
                tools=tools, #TODO add tools functionality
                text=response_format,
                timeout=TIMEOUT
            )
        return response

    def prompt_llm_response(self, prompt, system_prompt=None, messages = None, response_format=None, tools=None):
        start_time = time.time()
        print(f"{self.MODEL_NAME} thinking...")

        if self.local_model:
            messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
            response = self.vllm_response(messages=messages, response_format=response_format)
            output = response.choices[0].message.content
        else:
            response = self.chatGPT_response(
                input=prompt,
                system_prompt=system_prompt,
                tools=tools,
                response_format=response_format
            )
            output = response.output_text

        print(output)

        print()
        print("\nDone in", round(time.time() - start_time, 2), "seconds")

        return output

    
    def prompt_valid_DISARM_response(self, prompt, system_prompt, response_format):
        # ensure response is JSON
        result_raw = self.prompt_llm_response(prompt, system_prompt, response_format=response_format)
        try:
            result_parsed = json.loads(result_raw)       
        except json.JSONDecodeError as e:
            raise ValueError("Model failed to return valid JSON") from e
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

        if self.local_model:
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "tactics_schema",
                    "schema": ta_format
                    }
                }
        else:
            response_format={
                    "format": {
                        "type": "json_schema",
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
        ta_result_parsed = self.prompt_valid_DISARM_response(prompt=ta_prompt, system_prompt=system_prompt, response_format=response_format)
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
            "required": ["Techniques"],
            "additionalProperties": False
        }
        
        # alter the system prompt depending on mode
        if self.clf_mode == Mode.INVESTIGATE:
            system_prompt = T_SYS_INVEST
        else:
            system_prompt = T_SYS_RECOG

        if self.local_model:
            response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "techniques_schema",
                        "schema": t_format
                        }
                    }
        else:
            response_format={
                    "format": {
                            "type": "json_schema",
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
            t_result_parsed = self.prompt_valid_DISARM_response(prompt=t_prompt, system_prompt=system_prompt, response_format=response_format)
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

def test_interface():
    test_content = """
Bushra Bibi led a protest to free Imran Khan - what happened next is a mystery
Bushra Bibi, wife of jailed former Pakistani Prime Minister Imran Khan, and supporters of Khan's party Pakistan Tehreek-e-Insaf (PTI) attend a rally demanding his release, in Islamabad, Pakistan, November 26, 2024.
Image source,Reuters
Image caption,
Imran Khan's wife, Bushra Bibi, encouraged protesters into the heart of Pakistan's capital, Islamabad

ByFarhat Javed
BBC News, in Islamabad
Published
30 November 2024
A charred lorry, empty tear gas shells and posters of former Pakistan Prime Minister Imran Khan - it was all that remained of a massive protest led by Khan’s wife, Bushra Bibi, that had sent the entire capital into lockdown.

Just a day earlier, faith healer Bibi - wrapped in a white shawl, her face covered by a white veil - stood atop a shipping container on the edge of the city as thousands of her husband’s devoted followers waved flags and chanted slogans beneath her.

It was the latest protest to flare since Khan, the 72-year-old cricketing icon-turned-politician, was jailed more than a year ago after falling foul of the country's influential military which helped catapult him to power.

“My children and my brothers! You have to stand with me,” Bibi cried on Tuesday afternoon, her voice cutting through the deafening roar of the crowd.

“But even if you don’t,” she continued, “I will still stand firm.

“This is not just about my husband. It is about this country and its leader.”

It was, noted some watchers of Pakistani politics, her political debut.

But as the sun rose on Wednesday morning, there was no sign of Bibi, nor the thousands of protesters who had marched through the country to the heart of the capital, demanding the release of their jailed leader.

While other PMs have fallen out with Pakistan's military in the past, Khan's refusal to stay quiet behind bars is presenting an extraordinary challenge - escalating the standoff and leaving the country deeply divided.

Exactly what happened to the so-called “final march”, and Bibi, when the city went dark is still unclear.

All eyewitnesses like Samia* can say for certain is that the lights went out suddenly, plunging D Chowk, the square where they had gathered, into blackness.

Women and children collect recyclables from the burnt truck used by Bushra Bibi, wife of jailed former Pakistani Prime Minister Imran Khan. The truck is in the middle of a quiet main road
Image source,Reuters
Image caption,
Within a day of arriving, the protesters had scattered - leaving behind Bibi's burnt-out vehicle

As loud screams and clouds of tear gas blanketed the square, Samia describes holding her husband on the pavement, bloodied from a gun shot to his shoulder.

"Everyone was running for their lives," she later told BBC Urdu from a hospital in Islamabad, adding it was "like doomsday or a war".

"His blood was on my hands and the screams were unending.”

But how did the tide turn so suddenly and decisively?

Just hours earlier, protesters finally reached D Chowk late afternoon on Tuesday. They had overcome days of tear gas shelling and a maze of barricaded roads to get to the city centre.

Many of them were supporters and workers of the Pakistan Tehreek-e-Insaf (PTI), the party led by Khan.

He had called for the march from his jail cell, where he has been for more than a year on charges he says are politically motivated.

Now Bibi - his third wife, a woman who had been largely shrouded in mystery and out of public view since their unexpected wedding in 2018 - was leading the charge.

“We won’t go back until we have Khan with us,” she declared as the march reached D Chowk, deep in the heart of Islamabad’s government district.

Hundreds of people make their way along a highway with bushes on either side, and handful of cars in amongst the protesters. Some people hold giant red and green flags. Smoke can be seen rising in the distance.
Image source,Reuters
Image caption,
Thousands had marched for days to reach Islamabad, demanding former Prime Minister Imran Khan be released from jail

Insiders say even the choice of destination - a place where her husband had once led a successful sit in - was Bibi’s, made in the face of other party leader’s opposition, and appeals from the government to choose another gathering point.

Her being at the forefront may have come as a surprise. Bibi, only recently released from prison herself, is often described as private and apolitical. Little is known about her early life, apart from the fact she was a spiritual guide long before she met Khan. Her teachings, rooted in Sufi traditions, attracted many followers - including Khan himself.

Was she making her move into politics - or was her sudden appearance in the thick of it a tactical move to keep Imran Khan’s party afloat while he remains behind bars?

For critics, it was a move that clashed with Imran Khan’s oft-stated opposition to dynastic politics.

There wasn’t long to mull the possibilities.

After the lights went out, witnesses say that police started firing fresh rounds of tear gas at around 21:30 local time (16:30 GMT).

The crackdown was in full swing just over an hour later.

At some point, amid the chaos, Bushra Bibi left.

Videos on social media appeared to show her switching cars and leaving the scene. The BBC couldn’t verify the footage.

By the time the dust settled, her container had already been set on fire by unknown individuals.

By 01:00 authorities said all the protesters had fled.

Policemen stand guard at the Red Zone area after security forces conducted an overnight operation against the supporters of jailed former prime minister Imran Khan's Pakistan Tehreek-e-Insaf (PTI) party during a protest for the release of Imran Khan, early in Islamabad on November 27, 2024.
Image source,Getty Images
Image caption,
Security was tight in the city, and as night fell, lights were switched off - leaving many in the dark as to what exactly happened next

Eyewitnesses have described scenes of chaos, with tear gas fired and police rounding up protesters.

One, Amin Khan, said from behind an oxygen mask that he joined the march knowing that, "either I will bring back Imran Khan or I will be shot".

The authorities have have denied firing at the protesters. They also said some of the protesters were carrying firearms.

The BBC has seen hospital records recording patients with gunshot injuries.

However, government spokesperson Attaullah Tarar told the BBC that hospitals had denied receiving or treating gunshot wound victims.

He added that "all security personnel deployed on the ground have been forbidden" from having live ammunition during protests.

But one doctor told BBC Urdu that he had never done so many surgeries for gunshot wounds in a single night.

"Some of the injured came in such critical condition that we had to start surgery right away instead of waiting for anaesthesia," he said.

While there has been no official toll released, the BBC has confirmed with local hospitals that at least five people have died.

Police say at least 500 protesters were arrested that night and are being held in police stations. The PTI claims some people are missing.

And one person in particular hasn’t been seen in days: Bushra Bibi.

Municipal workers clean the street leading to Red Zone area next to damaged vehicles after an overnight security forces operation against the supporters of jailed former prime minister Imran Khan's Pakistan Tehreek-e-Insaf (PTI) party in Islamabad on November 27, 2024. 
Image source,Getty Images
Image caption,
The next morning, the protesters were gone - leaving behind just wrecked cars and smashed glass

“She abandoned us,” said one PTI supporter.

Others defended her. “It wasn’t her fault,” insisted another. “She was forced to leave by the party leaders.”

Political commentators have been more scathing.

“Her exit damaged her political career before it even started,” said Mehmal Sarfraz, a journalist and analyst.

But was that even what she wanted?

Khan has previously dismissed any thought his wife might have her own political ambitions - “she only conveys my messages,” he said in a statement attributed to him on his X account.

Bushra Bibi and Imran Khan are shielded by a white sheet as they arrive at a courthouse. Bibi has turned to look at the camera, her hair is covered so is her nose and mouth.
Image source,EPA
Image caption,
Imran Khan and Bushra Bibi, pictured here arriving at court in May 2023, married in 2018

Speaking to BBC Urdu, analyst Imtiaz Gul calls her participation “an extraordinary step in extraordinary circumstances".

Gul believes Bushra Bibi’s role today is only about “keeping the party and its workers active during Imran Khan’s absence”.

It is a feeling echoed by some PTI members, who believe she is “stepping in only because Khan trusts her deeply”.

Insiders, though, had often whispered that she was pulling the strings behind the scenes - advising her husband on political appointments and guiding high-stakes decisions during his tenure.

A more direct intervention came for the first time earlier this month, when she urged a meeting of PTI leaders to back Khan’s call for a rally.

Pakistan’s defence minister Khawaja Asif accused her of “opportunism”, claiming she sees “a future for herself as a political leader”.

But Asma Faiz, an associate professor of political science at Lahore University of Management Sciences, suspects the PTI’s leadership may have simply underestimated Bibi.

“It was assumed that there was an understanding that she is a non-political person, hence she will not be a threat,” she told the AFP news agency.

“However, the events of the last few days have shown a different side of Bushra Bibi.”

But it probably doesn’t matter what analysts and politicians think. Many PTI supporters still see her as their connection to Imran Khan. It was clear her presence was enough to electrify the base.

“She is the one who truly wants to get him out,” says Asim Ali, a resident of Islamabad. “I trust her. Absolutely!”
"""
    # llm = DISARM_LLM(
    #     article_content=test_content,
    #     model_name="gpt-5.6-luna",
    #     local_model=False,
    #     mode=Mode.RECOGNISE,
    #     check_sub_techniques=True
    # )

    llm = DISARM_LLM(
        article_content=test_content,
        model_name="google/gemma-4-26B-A4B-it",
        local_model=True,
        mode=Mode.INVESTIGATE,
        check_sub_techniques=True
    )

    _, techniques = llm.batch_clf()

    print(techniques)

if __name__ == "__main__":
    test_interface()