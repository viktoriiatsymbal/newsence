from datasets import load_dataset
# not used anymore, was used when we had ag news dataset
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def load_dataset_local(name: str = "ag_news", split: str = "train[:5000]"):
    print(f"Loading dataset: {name}")
    ds = load_dataset(name, split=split)
    ds = ds.map(lambda x: {"category": LABEL_MAP[int(x["label"])]})
    return ds

