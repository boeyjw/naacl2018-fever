import argparse
import json

from tqdm import tqdm
from pathlib import Path

from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from retrieval.filter_uninformative import uninformative

parser = argparse.ArgumentParser()
parser.add_argument('db_path', type=str, help='/path/to/fever.db')
parser.add_argument('datain', type=str, help='/path/to/data-train-dev-test; Filename must have train, test or dev words!')
parser.add_argument('dataout', type=str, help='/path/to/output')

args = parser.parse_args()

datain = Path(args.datain)
datain_d = {
    "train": list(datain.glob("*train*.jsonl")), 
    "dev": list(datain.glob("*dev*.jsonl")), 
    "test": list(datain.glob("*test*.jsonl"))
}

jlr = JSONLineReader()

docdb = FeverDocDB(args.db_path)

idx = docdb.get_non_empty_doc_ids()
idx = list(filter(lambda item: not uninformative(item),tqdm(idx)))


r = SimpleRandom.get_instance()

if datain_d["train"]:
    with open("{0}/{1}.ns.rand.jsonl".format(args.dataout, datain_d["train"][0].stem), "w+") as f:
        for line in jlr.read(str(datain_d["train"][0])):
            if line["label"] == "NOT ENOUGH INFO":

                for evidence_group in line['evidence']:
                    for evidence in evidence_group:
                        evidence[2] = idx[r.next_rand(0, len(idx))]
                        evidence[3] = -1


            f.write(json.dumps(line)+"\n")

if datain_d["dev"]:
    with open("{0}/{1}.ns.rand.jsonl".format(args.dataout, datain_d["dev"][0].stem), "w+") as f:
        for line in jlr.read(str(datain_d["dev"][0])):
            if line["label"]=="NOT ENOUGH INFO":
                for evidence_group in line['evidence']:
                    for evidence in evidence_group:
                        evidence[2] = idx[r.next_rand(0, len(idx))]
                        evidence[3] = -1

            f.write(json.dumps(line)+"\n")


if datain_d["test"]:
    with open("{0}/{1}.ns.rand.jsonl".format(args.dataout, datain_d["test"][0].stem), "w+") as f:
        for line in jlr.read(str(datain_d["test"][0])):
            if line["label"] == "NOT ENOUGH INFO":
                for evidence_group in line['evidence']:
                    for evidence in evidence_group:
                        evidence[2] = idx[r.next_rand(0, len(idx))]
                        evidence[3] = -1

            f.write(json.dumps(line)+"\n")
