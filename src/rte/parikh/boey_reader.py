from typing import Dict
from pathlib import Path
import json
import logging
import os
import gzip
import pickle as pkl

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
# from allennlp.data.dataset import Dataset
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("fever")
class FEVERReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 wiki_tokenizer: Tokenizer = None,
                 claim_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._wiki_tokenizer = wiki_tokenizer or WordTokenizer()
        self._claim_tokenizer = claim_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self._ID2LABEL = {0: "SUPPORTS", 1: "NOT ENOUGH INFO", 2: "REFUTES"}

    def read_data(self, fp):
        fn = gzip.open(fp, "rb") if fp.suffix == ".gz" else open(fp, "r")
        fp_str = str(fp)
        try:
            if ".jsonl" in os.path.basename(fp_str):
                data = [json.loads(l.decode("utf8") if fp.suffix == ".gz" else l) for l in fn.readlines()]
            elif ".json" in os.path.basename(fp_str):
                data = json.loads(fn.read())
            elif ".pkl.gz" in os.path.basename(fp_str):
                data = pkl.loads(fn.read())
            else:
                raise NotImplementedError(f"{os.path.basename(fp)} suffix is unsupported.")
        finally:
            fn.close()
        return data

    @overrides
    def read(self, file_path: str):
        instances = []

        ds = self.read_data(Path(file_path))

        for instance in tqdm.tqdm(ds):
            if instance is None:
                continue

            premise = instance["evidence"]
            if len(premise.strip()) == 0:
                premise = ""

            hypothesis = instance["claim"]
            label = self._ID2LABEL[instance["labels"]]
            instances.append(self.text_to_instance(premise, hypothesis, label))
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Batch(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._wiki_tokenizer.tokenize(premise) if premise is not None else None
        hypothesis_tokens = self._claim_tokenizer.tokenize(hypothesis)
        fields['premise'] = TextField(premise_tokens, self._token_indexers) if premise is not None else None
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'FEVERReader':
        claim_tokenizer = Tokenizer.from_params(params.pop('claim_tokenizer', {}))
        wiki_tokenizer = Tokenizer.from_params(params.pop('wiki_tokenizer', {}))

        # token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        # Hack as TokenIndexer does not have "dict_from_params" function
        token_indexers = {'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}
        params.assert_empty(cls.__name__)
        return FEVERReader(claim_tokenizer=claim_tokenizer,
                           wiki_tokenizer=wiki_tokenizer,
                           token_indexers=token_indexers)

