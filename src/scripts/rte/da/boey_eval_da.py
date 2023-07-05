import os

from copy import deepcopy
from typing import List, Union, Dict, Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#from allennlp.common.util import prepare_environment
from allennlp.data import Tokenizer
from allennlp.models import load_archive
from common.util.log_helper import LogHelper
from rte.parikh.boey_reader import FEVERReader
from tqdm import tqdm
import argparse
import logging
import json
import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def eval_model(args):
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)

    config = archive.config
    ds_params = config["dataset_reader"]

    model = archive.model
    model.eval()

    # COMMENT: token_indexers = None uses the default SingleID Tokenizer
    reader = FEVERReader(wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                        claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                        token_indexers=None)

    logger.info("Reading training data from %s", args.in_file)
    data = reader.read(args.in_file).instances

    actual = []
    predicted = []

    if args.log is not None:
        f = open(args.log,"w+")

    for item in tqdm(data):
        prediction = model.forward_on_instance(item)
        cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]

        if "label" in item.fields:
            actual.append(item.fields["label"].label)
        predicted.append(cls)

        if args.log is not None:
            if "label" in item.fields:
                f.write(json.dumps({"predicted_logits":prediction["label_logits"].tolist(), "predicted":cls, "predicted_proba": prediction["label_probs"].tolist()})+"\n")
            else:
                f.write(json.dumps({"predicted":cls})+"\n")

    if args.log is not None:
        f.close()


    # if len(actual) > 0:
    #     print(accuracy_score(actual, predicted))
    #     print(classification_report(actual, predicted))
    #     print(confusion_matrix(actual, predicted))

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    LogHelper.get_logger(__name__)


    parser = argparse.ArgumentParser()

    parser.add_argument('archive_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('in_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--log', required=False, default=None,  type=str, help='/path/to/saved/db.db')

    parser.add_argument("--cuda-device", type=int, default=0, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')



    args = parser.parse_args()
    eval_model(args)
