import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from loguru import logger
from DISARM_LLM_Interface import DISARM_LLM, Mode
import json
import contextlib


DATASET_PATH = os.path.join(".data/euvsdisinfo.csv")
SUBSET_PATH = ".data/euvsdisinfo_bal_set.csv"
CACHE_PATH = os.path.join(".data/euvsdisinfo_cache.json")


def ballance_dataset(df, required_balance, min_class_samples):
    # for each language, ballance the number of articles that are trustworthy and disinfo by a balance threshold
    languages = df["language"].unique()
    df_list = []
    for language in languages:
        language_df = df[df["language"] == language]
        disinfo_df = language_df[language_df["class"] == 1]
        trustworthy_df = language_df[language_df["class"] == 0]

        disinfo_sample_size = len(disinfo_df)
        trustworthy_sample_size = len(trustworthy_df)

        # balancee to match the balance threshold
        if len(disinfo_df) / len(language_df) > required_balance: 
            disinfo_sample_size = int(len(trustworthy_df) * required_balance / (1 - required_balance))
        elif len(trustworthy_df) / len(language_df) > required_balance:
            trustworthy_sample_size = int(len(disinfo_df) * required_balance / (1 - required_balance))

        # if sample size is less than minimum samples, remove the language from the dataset
        if min(disinfo_sample_size, trustworthy_sample_size) < min_class_samples:
            continue

        disinfo_df = disinfo_df.sample(n=disinfo_sample_size, random_state=42)
        trustworthy_df = trustworthy_df.sample(n=trustworthy_sample_size, random_state=42)

        df_list.append(pd.concat([disinfo_df, trustworthy_df]))
    df = pd.concat(df_list).reset_index(drop=True)
    return df
    

def create_euvsdisinfo_bal_set():
    """Selects a subset of the euvsdisinfo database that is ballanced between disinfo and trustworthy
    articles between each language and saves it to .data/euvsdisinfo_bal_set"""

    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df = df.fillna("")
        df["text"] = df.apply(
            lambda x: x["article_title"] + " " + x["article_text"] if x["article_title"] != "" else x["article_text"],
            axis=1,
        )  # combine the title and text into a single column
        df["language"] = df["article_language"]
        df["stratify"] = df["class"] + df["language"]
        df["class"] = df["class"].apply(lambda x: 1 if x == "disinformation" else 0)
        df["label"] = df["class"]
        df["disarm_techniques"] = None

        df = ballance_dataset(df, required_balance=0.6, min_class_samples=2)

        # save the test set to .data/euvsdisinfo_bal_set
        df.to_csv(SUBSET_PATH, index=False)

        # draw a bar chart of the number of articles per language and class with plt
        df.groupby(["language", "class"]).size().unstack(fill_value=0).plot(kind="bar", figsize=(15, 6))
        plt.title("Distribution of Articles by Language and Class")
        plt.xlabel("Language")
        plt.ylabel("Number of Articles")
        plt.legend(["Trustworthy", "Disinformation"])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(".data/euvsdisinfo_bal_set_distribution.png")

        # print the number of articles per language and class
        print(df.groupby(["language", "class"]).size().unstack(fill_value=0))
        print(f"Total number of articles: {len(df)}")
    else:
        raise Exception(f"{DATASET_PATH} does not exist. Please download the dataset first.")


def load_cache(path=CACHE_PATH):
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

def main():
    """Uses the LLM interface to label the articles with DISARM techniques"""
    if not os.path.exists(SUBSET_PATH):
        create_euvsdisinfo_bal_set()

    df = pd.read_csv(SUBSET_PATH)
    skipped = 0
    failures = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # break if it has reached the desired size
        cache = load_cache()
        if len(cache) >= len(df):
            logger.debug("Cache Reached desired length, stopping")
            break
        # check if already classified in cache
        if row.article_id in [c["article_id"] for c in cache]:
            skipped+=1
            logger.debug(f"Skipped: {skipped}")
            continue

        # classify the article
        with open(os.devnull, 'w') as f:
            # ignore print statements from the disarm classifier
            with contextlib.redirect_stdout(f):
                llm = DISARM_LLM(
                    article_content=row.text,
                    model_name="google/gemma-4-26B-A4B-it",
                    local_model=True,
                    mode=Mode.INVESTIGATE,
                    check_sub_techniques=False,
                    debug_log=False,
                    web_search=True,
                )
                try:
                    _, techniques = llm.batch_clf()
                    logger.debug(
                        f"Successfully labelled {row.article_id} with techniques - {techniques}"
                    )
                except Exception as e:
                    logger.error("ERROR: ", e)
                    failures+=1
                    continue
        data = dict(row)
        data["disarm_techniques"] = techniques
        with open(CACHE_PATH, "a") as f:
            f.write(json.dumps(data) + "\n")
    
    df = load_cache(CACHE_PATH)
    df = pd.DataFrame(df)
    df.to_csv(".data/euvsdisinfo_labelled.csv", index=False)

    # SPLITTING INTO TRAIN AND TEST DATA
    train_df, dev_df = train_test_split(df, test_size=0.1, stratify=df["stratify"], random_state=42)

    train_df = train_df[["text", "disarm_techniques", "label", "language", "keywords", "debunk_date"]]
    dev_df = dev_df[["text", "disarm_techniques", "label", "language", "keywords", "debunk_date"]]

    train_df.to_csv(".data/experiments/euvsdisinfo.csv", index=False)
    dev_df.to_csv(".data/experiments/euvsdisinfo_dev.csv", index=False)

if __name__ == "__main__":
    main()