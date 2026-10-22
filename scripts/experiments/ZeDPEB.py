# Benchmarking software for the DISARM Interpreter

# for every incident file:
    # get the required techniques and tactics for the benchmark
    # Identify tactics with model
    # check if tactics identified contain required ones
    # For ONLY the tactics relevant to that benchmark, get the techniques
    # Check the identified techniques contain the required ones

import time
import json
import os
import sys
import requests
from bs4 import BeautifulSoup
from enum import Enum
from suffix_trees import STree
from urllib.parse import urlencode
import numpy as np
import re
import uuid
import argparse

from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DISARM_DATA_MASTER import DISARMDataMaster, get_mitre_external_id
from DISARM_LLM_Interface import DISARM_LLM
from pathlib import Path

class ClfMode(Enum):
    SELECT_ALL = "SelectAll"
    BATCH_FAST = "FastBatch"
    BATCH_FULL = "FullBatch"
    SINGLE = "Single"

JSON_DATA = Path(__file__).parent.parent / ".data" / "DISARM.json"

# with help from https://github.com/joaoaleite/euvsdisinfo/blob/main/scripts/collect/collect.py

def load_cache(path= Path(__file__).parent.parent.parent / ".data" / "incidents.json"):
    cache = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                try:
                    cache.append(json.loads(line))
                except json.decoder.JSONDecodeError:
                    print("Wrong formatting at line", i + 1)

    else:
        print("Cache file does not exist. Creating a new one.")
        with open(path, "w") as f:
            pass

    return cache

def get_related_tactics(technique_external_id):
    if type(technique_external_id) == list:
        article_tactics = []
        for tech in technique_external_id:
            tac = get_related_tactics(tech)
            if tac not in article_tactics:
                article_tactics.append(tac)
        return article_tactics
    else:
        with open(JSON_DATA, "r", encoding="utf-8") as f:
            disarm_json = json.load(f)
            for obj in disarm_json["objects"]:
                if obj.get("type") == "attack-pattern":
                    if get_mitre_external_id(obj) == technique_external_id:
                        for phase in obj.get("kill_chain_phases", []):
                            return phase.get("phase_name")
            return None
    

# retrieve the incident data from the disarm xlsx file
def retrieve_incident_data(disarm_data :DISARMDataMaster, incident_id, api_token):
    #fetch incident data
    article_techniques = disarm_data.get_incident_techniques(incident_id)
    #break if no techniques
    if len(article_techniques) <= 0: return None
    article_tactics = get_related_tactics(article_techniques)
    #fetch the incident url content
    urls = disarm_data.get_incident_urls(incident_id)
    if urls == []: 
        return None
    elif len(urls) > 1:
        print_log("Warning: Multiple urls for incident: ", incident_id)
    article_content = None
    print("Fetching Article Content")

    # Lines 96-107 done with help from https://github.com/joaoaleite/euvsdisinfo/blob/main/scripts/collect/collect.py
    params = urlencode({"token": api_token, "url": urls[0], "timeout": 1000 * 60})
    diffbot_url = f"https://api.diffbot.com/v3/article?{params}"

    # Make the GET request to the Diffbot Article API
    diffbot_response = requests.get(diffbot_url, headers={"User-Agent": "Mozilla/5.0"})
    if diffbot_response.status_code == 200:
        diffbot_data = json.loads(diffbot_response.content)
        # check diffbot has returned content, if not, fallback to standard http requests
        if "objects" in diffbot_data:
            article_content = diffbot_data["objects"][0].get("title")
            article_content += diffbot_data["objects"][0]["text"]
        else:
            response = requests.get(url=urls[0], headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = soup.find_all("p")
                article_content = "\n".join(p.get_text() for p in paragraphs)
            else:
                return None
        print(f"Done. Fetched article of length {len(article_content)}")
        print_log(f"{'-'*50}\nRetrieved article from {urls[0]}\n")
        data = dict(incident_id=incident_id, article_content=article_content, article_tactics=article_tactics, article_techniques=article_techniques)
        with open(".data/incidents.json", "a") as f:
            f.write(json.dumps(data) + "\n")
    else:
        print_log(f"\nError when attempting to fetch incident url content. Code: {diffbot_response.status_code}\n")
        return None
    
    return article_content, article_tactics, article_techniques
    
def log_to_file(message):
    # retrieve the execution log
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(message)

def new_log_file(mode, model, enbl_precision, enbl_rationale, num_tests):
    global log_filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = Path(__file__).parent / f".ZeDPEB_logs/log_{timestamp}{"_RAT" if enbl_rationale else ""}_{mode.value}_{num_tests if num_tests>0 else "COMPLETE"}.txt"
    log_to_file(f"ZeDPEB Log File - Created on {timestamp}\n\n{'='*50}")
    setup_str = f"""
SETUP
mode  - {mode}
model - {model}
test  - Recall {"& Precision" if enbl_precision else ""} {"& Rationale" if enbl_rationale else ""}
{'='*50}
"""
    log_to_file(setup_str)

# gets the part of the technique string that prepends the '.' 
def technique_partial(technique):
    return technique.split('.')[0]

def partial_match(correct_technique, identified_techniques):
    for identified_technique in identified_techniques:
        if technique_partial(correct_technique) == technique_partial(identified_technique):
            return True
    return False

def calc_precision(n_pos, n_false):
    try:
        p = n_pos / (n_pos + n_false)
    except ZeroDivisionError:
        return 0
    return p

def calc_f1(precision, recall):
    return (2 * precision * recall) / (precision + recall)

def test_rationale(technique_positives, identified_techniques, incident_id, disarm_llm: DISARM_LLM, mode):
    print_log(f"\nRationale Test for incident {incident_id}\n")
    # get rationales with faithfullness checks
    model_rationales, comprehensiveness, sufficiency = get_model_rationales(identified_techniques, technique_positives, disarm_llm, mode)
    ave_recall, ave_precision, f1 = rationale_plausability(model_rationales, incident_id)
    log_rationale_stats(len(model_rationales), comprehensiveness, sufficiency, ave_recall, ave_precision, f1)
    return comprehensiveness, sufficiency, ave_recall, ave_precision, f1, len(model_rationales)

# TODO add methods for each mode
def get_model_rationales(identified_techniques, technique_positives, disarm_llm: DISARM_LLM, mode):
    # extracts the models rationales and checks how faithfull they are
    # depends on classificaton mode
    if mode == ClfMode.SINGLE:
        rationales = []
        for id in technique_positives:
            rationales.extend(disarm_llm.identify_rationales([id]))
    elif mode == ClfMode.BATCH_FAST or mode == ClfMode.BATCH_FULL:
        rationales = []
        tactics = get_related_tactics(technique_positives)
        for t in tactics:
            batch_ids = []
            for id in identified_techniques:
                if t in get_related_tactics(id):
                    batch_ids.append(id)
            rationales.extend(disarm_llm.identify_rationales(batch_ids))
    else:
        rationales = disarm_llm.identify_rationales(identified_techniques)
            
    article_content = disarm_llm.article_content
    sufficiency = rationale_sufficiency(rationales, disarm_llm, mode) # to change
    comprehensiveess = rationale_comprehensiveness(article_content, rationales, disarm_llm, mode) # to change
    
    return rationales, comprehensiveess, sufficiency

def log_rationale_stats(n_tests, comp, suff, recall, prec, f1):
    log = f"""{'.'*50}
Rationale test results:
    Faithfullness:
        Comprehensiveness: {comp}/{n_tests}
        Sufficiency:       {suff}/{n_tests}
    Plausability (Rouge Scores):
        Recall:    {recall}
        Precision: {prec} 
        F1:        {f1}
{'.'*50}"""
    print_log(log)
    
# Faithfullness metrics
def rationale_comprehensiveness(article_content: str, model_rationales, disarm_llm: DISARM_LLM, mode):
    # check if removing model rationale from the model input removes the classification 
    required_classifications = []
    for rationale in model_rationales:
        label, quotes = rationale
        required_classifications.append(label)
        for i, quote in enumerate(quotes):
            # remove the largest common substring from the article content
            print(f"Quote {i}: {repr(quote)}")
            while True:
                lcs = STree.STree([quote, article_content]).lcs()
                if not lcs:
                    break
                if len(lcs) < 5:
                    break
                article_content = article_content.replace(lcs, "", -1)
                print(f"Replaced {repr(lcs)} in article content")
                
    # reclassify (using a unique id at the front to ensure no prefix caching)
    disarm_llm.article_content = f"ID: {uuid.uuid4()} " + article_content
    match mode:
        case ClfMode.SELECT_ALL:
            identified_techniques = disarm_llm.select_all_clf()
        case ClfMode.BATCH_FAST:
            _, identified_techniques = disarm_llm.batch_clf(fast=True)
        case ClfMode.BATCH_FULL:
            identified_techniques = reduced_full_batch_clf(article_tactics=get_related_tactics(required_classifications), disarm_llm=disarm_llm)
        case ClfMode.SINGLE:
            identified_techniques = reduced_single_clf(required_classifications, disarm_llm)
    # measure difference
    diff = 0
    for technique in required_classifications:
        if technique not in identified_techniques:
            diff += 1
    return diff

            

def rationale_sufficiency(model_rationales, disarm_llm: DISARM_LLM, mode):
    # check if removing everything except the model rationale from the model input keeps the classification
    required_classifications = []
    article_content = ""
    for rationale in model_rationales:
        label, quotes = rationale
        required_classifications.append(label)
        for quote in quotes:
            if quote not in article_content:
                article_content += quote
    # reclassify
    disarm_llm.article_content = article_content
    match mode:
        case ClfMode.SELECT_ALL:
            identified_techniques = disarm_llm.select_all_clf()
        case ClfMode.BATCH_FAST:
            _, identified_techniques = disarm_llm.batch_clf(fast=True)
        case ClfMode.BATCH_FULL:
            identified_techniques = reduced_full_batch_clf(article_tactics=get_related_tactics(required_classifications), disarm_llm=disarm_llm)
        case ClfMode.SINGLE:
            identified_techniques = reduced_single_clf(required_classifications, disarm_llm)
    # measure difference
    match = 0
    for technique in required_classifications:
        if technique in identified_techniques:
            match += 1
    return match

def rationale_plausability(model_rationales, incident_id):
    # actually scores how similar the models rationale is to the 'gold' rationale
    total_recall = []
    total_precision = []
    total_f1 = []
    print_log(f"\nRationale Plausability test for incident: {incident_id}")
    # returns a list of tuples (technique, quotes)
    gold_rationales = get_gold_rationales(incident_id=incident_id)
    for gold in gold_rationales:
        technique, gold_quotes = gold
        # get the model rationale for the associated gold technique rationale
        model = next((m for m in model_rationales if m[0] == technique), None)
        if model is None:
            # skip if not identified by model
            continue
        _, model_quotes = model
        recall, precision, f1 = rationale_rouge(model_quotes, gold_quotes)
        log = f"""
Rationale score for technique: {technique}
    Recall: {recall}
    Precision: {precision}
    F1: {f1}
"""
        print_log(log)
        total_recall.append(recall)
        total_precision.append(precision)
        total_f1.append(f1)

    #calculate average scores
    ave_recall = np.average(total_recall)
    ave_precision = np.average(total_precision)
    ave_f1 = np.average(total_f1)

    return ave_recall, ave_precision, ave_f1

def get_gold_rationales(incident_id):
    incident_data = DISARMDataMaster().get_incident_techniques_with_desc(incidentid=incident_id)
    rationales = []
    for d in incident_data:
        #split descriptions into quotes
        (external_id, desc) = d
        quotes = re.findall(r"<i>(.*?)</i>", desc, re.IGNORECASE)
        if quotes == []:
            quotes = re.findall(r"\_(.*?)\_", desc, re.IGNORECASE)
        rationales.append((external_id, quotes))
    return rationales

def normalise_rationale(quotes):
    full_rationale = ""
    for q in quotes:
        full_rationale += q
    return full_rationale

def rationale_rouge(model_quotes, gold_quotes):
    model_quotes = normalise_rationale(model_quotes)
    gold_quotes = normalise_rationale(gold_quotes)

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(model_quotes, gold_quotes)

    return scores['rouge1']

def reduced_full_batch_clf(article_tactics, disarm_llm: DISARM_LLM):
    identified_techniques = []
    for required_tactic in article_tactics:
        # perform technique classifications (ONLY for relevant tactics, no need to do full batchClf):
        identified_techniques += disarm_llm.identify_techniques_for_tactic(required_tactic)
    return identified_techniques

def reduced_single_clf(article_techniques, disarm_llm: DISARM_LLM):
    identified_techniques = []
    for required_technique in article_techniques:
        identified_techniques += disarm_llm.identify_techniques(filter_ids=[required_technique]) 
    return identified_techniques

# TODO: implement more statistics
def test_clf(model, api_token, mode=ClfMode.BATCH_FULL, num_tests=-1, checkpoint=0, skip_to_id=0, enbl_precision=True, enbl_rationale=False):
    i_cache = load_cache()
    is_fast_batch = mode == ClfMode.BATCH_FAST 

    start_time = time.time()
    disarm_data = DISARMDataMaster()
    new_log_file(mode, model, enbl_precision, enbl_rationale, num_tests)

    incident_ids = disarm_data.get_incident_ids()
    # variables to keep track of overall performance
    tot_n_rat_tests = 0
    ta_total = t_total = ta_mtchs_total = t_mtchs_total = t_abs_mtchs_total = t_prt_mtchs_total = 0
    ta_false_total = t_false_total = t_abs_false_total = 0
    #rationales
    r_comp_total = r_suff_total = 0
    r_rec = [] 
    r_prec = [] 
    r_f1 = []

    failures = []
    
    # incident loop
    for i, incident_id in enumerate(incident_ids):
        try:
            if i+1 < skip_to_id:
                continue
            # break loop if number of tests reached
            if num_tests == 0:
                break
            tactic_positives = 0
            technique_abs_positives = 0
            technique_abs_positives_list = []
            technique_partial_positives = 0

            # get the incident data
            if incident_id in [c["incident_id"] for c in i_cache]:
                print_log(f"\nRetrieved incident {incident_id} from cache\n")
                row = next(c for c in i_cache if c["incident_id"] == incident_id)
                article_content = row["article_content"]
                article_tactics = row["article_tactics"]
                article_techniques = row["article_techniques"]
                result = (article_content, article_tactics, article_techniques)
                
            else:
                result = retrieve_incident_data(disarm_data, incident_id, api_token)

            # skip if no url/data is available
            if result is None:
                print_log(f"Skipping incident {incident_id} - missing data\n")
                continue
            article_content, article_tactics, article_techniques = result
            if len(article_content) < 100:
                print_log(f"Skipping incident {incident_id} - insignificant amount of article content retrieved: ({len(article_content)})\n")
                continue
            if len(article_content) > 50000:
                print_log(f"Skipping incident {incident_id} - excessive amount of article content retrieved: ({len(article_content)})\n")
                continue
            if checkpoint > 0:
                print_log(f"Skipping incident {incident_id} by user request\n")
                checkpoint -= 1
                continue

            ta_total += len(article_tactics)
            t_total += len(article_techniques)
            
            disarm_llm = DISARM_LLM(
                article_content=article_content, 
                model_name=model, 
                local_model=True,
                mode=2,
                check_sub_techniques=True,
                debug_log=False,
                web_search=False
            )
            print_log(f"\nCurrent time taken: {round((time.time() - start_time)/60, 2)} minuites")
            print_log(f"\nTesting for incident with ID: {incident_id} \n{f"Required Tactics: {article_tactics}\n" if is_fast_batch else ""}Required Techniques: {article_techniques}")
            print_log(f"\nArticle Character Length: {len(article_content)}\n")

            identified_techniques = []
            identified_tactics = []

            if mode == ClfMode.BATCH_FAST or mode == ClfMode.BATCH_FULL:
                if enbl_precision or mode == ClfMode.BATCH_FAST:
                    identified_tactics, identified_techniques = disarm_llm.batch_clf(mode == ClfMode.BATCH_FAST)
                    for required_tactic in article_tactics:
                        if required_tactic in identified_tactics:
                            tactic_positives+=1
                else: # if precision is dissabled and doing full batch, then unnessasary batches can be ignored 
                    identified_techniques = reduced_full_batch_clf(article_tactics, disarm_llm)
            elif mode == ClfMode.SELECT_ALL:
                identified_techniques = disarm_llm.select_all_clf()
            elif mode == ClfMode.SINGLE:
                if enbl_precision:
                    identified_techniques = disarm_llm.single_clf()
                else:
                    # only check relevant techniques, no need to do full clf if precision is dissabled
                    identified_techniques = reduced_single_clf(article_techniques, disarm_llm)
                    
            print(f"Techniques identified by model: {identified_techniques}")
            for required_technique in article_techniques:
                if required_technique in identified_techniques:
                    technique_abs_positives+=1
                    technique_abs_positives_list.append(required_technique)
                elif partial_match(required_technique, identified_techniques):
                    technique_partial_positives+=1
            # record results
            technique_positives = technique_abs_positives + technique_partial_positives
            ta_false = len(identified_tactics) - tactic_positives
            t_false = len(identified_techniques) - (technique_positives)
            t_abs_false = len(identified_techniques) - (technique_abs_positives)
            # update totals
            ta_mtchs_total += tactic_positives
            t_mtchs_total += technique_positives
            t_abs_mtchs_total += technique_abs_positives
            t_prt_mtchs_total += technique_partial_positives

            ta_false_total += ta_false
            t_false_total += t_false
            t_abs_false_total += t_abs_false

            results_log = f"""Results:
    {f"Identified tactics({len(identified_tactics)}): {identified_tactics}" if is_fast_batch else ""}
    Identified techniques({(len(identified_techniques))}): {identified_techniques}

    Recall:
    {f"Correct Tactics: {tactic_positives}/{len(article_tactics)}" if is_fast_batch else ""}
    Correct Techniques: {technique_positives}/{len(article_techniques)}
        Absolute technique matches: {technique_abs_positives}
        Partial technique matches: {technique_partial_positives}
    """
            if enbl_precision:
                results_log += f"""Precision:
    {f"Tactics: {calc_precision(tactic_positives, ta_false)}" if is_fast_batch else ""}
    Techniques: {calc_precision(technique_positives, t_false)}
        Absolute Techniques: {calc_precision(technique_abs_positives, t_abs_false)}
    """
            print_log(results_log)
            
            if enbl_rationale and technique_abs_positives > 0:
                # test and log model rationale
                c, s, r, p, f, n_r_t = test_rationale(technique_abs_positives_list, identified_techniques, incident_id, disarm_llm, mode) 
                tot_n_rat_tests += n_r_t
                r_comp_total += c
                r_suff_total += s
                r_rec.append(r)
                r_prec.append(p)
                r_f1.append(f)
            elif technique_abs_positives <= 0:
                print_log(f"\nUnable to test for rationales - no absolute technique matches\n")

        except Exception as n_w:
            print(n_w)
            print_log(f"\nFailed at incident with id {incident_id}: {type(n_w)}, {n_w}\n")
            failures.append((incident_id, n_w))
        print_log('-'*50)
        num_tests-=1
            
    final_log = f"""
{'='*50}
Final Results:
{'-'*50}
Recall:
{f"Correct Tactics: {ta_mtchs_total}/{ta_total} \t ({(ta_mtchs_total/ta_total)*100}%)" if is_fast_batch else ""}
Correct Techniques: {t_mtchs_total}/{t_total} \t ({((t_mtchs_total)/t_total)*100}%)
    Absolute technique matches: {t_abs_mtchs_total} \t ({(t_abs_mtchs_total/t_total)*100}%)
    Partial technique matches: {t_prt_mtchs_total} \t ({(t_prt_mtchs_total/t_total)*100}%)
"""
    if enbl_precision:
        final_log += f"""{'-'*50}
Precision:
{f"Tactics: {calc_precision(ta_mtchs_total, ta_false_total)}" if is_fast_batch else ""}
Techniques: {calc_precision(t_mtchs_total, t_false_total)}
    Absolute Techniques: {calc_precision(t_abs_mtchs_total, t_abs_false_total)}
"""
    if enbl_rationale:
        final_log += f"""{'-'*50}
Rationale test results:
    Faithfullness:
        Comprehensiveness: {(r_comp_total/tot_n_rat_tests)*100}%
        Sufficiency:       {(r_suff_total/tot_n_rat_tests)*100}%
    Plausability (Rouge Scores):
        Recall:    {np.average(r_rec)}
        Precision: {np.average(r_prec)} 
        F1:        {np.average(r_f1)}
{'-'*50}"""
    if len(failures) > 0:
        final_log += f"\n{len(failures)} Failures:\n"
        for warn_id, n_w in failures:
            final_log += f"Failure at {warn_id}: {type(n_w)}, {n_w}\n"
    final_log += f"""
Time Taken: {round((time.time() - start_time)/60, 2)} minuites
{'='*50}"""
    print_log(final_log)

def print_log(str):
    print(str)
    log_to_file(str)

def main():
    diffbot_token = os.environ.get("DIFFBOT_TOKEN")

    # large_model = "Qwen/Qwen3.5-35B-A3B"
    # large_model = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    large_model = "google/gemma-4-26B-A4B-it"
    # large_model = "Qwen/QwQ-32B"
    # large_model = "google/gemma-4-E4B-it"
    # large_model = "Qwen/Qwen3.6-27B"
    
    enbl_rationale = False
    # for enbl_rationale in [False, True]:
    for mode in [ClfMode.SELECT_ALL, ClfMode.BATCH_FULL, ClfMode.BATCH_FAST]:
        test_clf(model=large_model, 
                api_token=diffbot_token, 
                mode=mode, 
                num_tests=-1, 
                checkpoint=0, 
                skip_to_id=0, 
                enbl_precision=not enbl_rationale, 
                enbl_rationale=enbl_rationale
                )
    # do not easure precicion due to intractable run time
    test_clf(model=large_model, api_token=diffbot_token, mode=ClfMode.SINGLE, num_tests=-1, checkpoint=0, skip_to_id=0, enbl_precision=False, enbl_rationale=enbl_rationale)
    # rationale test
    enbl_rationale = True
    for mode in [ClfMode.SINGLE, ClfMode.BATCH_FULL, ClfMode.SELECT_ALL]:
        test_clf(model=large_model, 
                api_token=diffbot_token, 
                mode=mode, 
                num_tests=-1, 
                checkpoint=0, 
                skip_to_id=0, 
                enbl_precision=not enbl_rationale, 
                enbl_rationale=enbl_rationale
                )
        
if __name__ == "__main__":
    # main()
    test_clf(
        model="google/gemma-4-26B-A4B-it",
        api_token=os.environ.get("DIFFBOT_TOKEN"),
        mode= ClfMode.BATCH_FULL,
        num_tests=1,
        enbl_precision=True,
        enbl_rationale=False
    )
