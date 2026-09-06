"""Register the scispaCy tokenizer used to build the AIR-MS DocBins.

``spacy train`` builds a fresh English tokenizer from the config, but the
DocBins were tokenized by scispaCy. Mismatched tokenization does not raise; it
silently degrades training because entity boundaries falling inside a token
cannot be learned. The base model is therefore loaded with all pipeline
components excluded so that only its tokenizer is materialized.
"""

import spacy
from spacy.language import Language
from spacy.util import registry


@registry.callbacks("airms.scispacy_tokenizer.v1")
def make_scispacy_tokenizer_callback():
    """Create an after-creation callback that installs scispaCy's tokenizer."""

    def copy_scispacy_tokenizer(nlp: Language) -> Language:
        sci = spacy.load(
            "en_core_sci_sm",
            exclude=[
                "tok2vec",
                "tagger",
                "attribute_ruler",
                "lemmatizer",
                "parser",
                "ner",
            ],
        )
        nlp.tokenizer.from_bytes(sci.tokenizer.to_bytes())
        return nlp

    return copy_scispacy_tokenizer
