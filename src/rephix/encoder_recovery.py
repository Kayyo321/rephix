"""Resolve the native task wrapper for a local encoder checkpoint."""

ENCODER_MODEL_TYPES = frozenset({"albert", "bert", "deberta-v2", "distilbert", "roberta"})


def encoder_auto_model(config, *, architecture=None):
    """Select a built-in Transformers AutoModel class from an exact architecture name."""
    from transformers.models.auto import modeling_auto as auto

    model_type = config.model_type
    if model_type not in ENCODER_MODEL_TYPES or config.is_decoder:
        raise ValueError(f"encoder task recovery is unsupported for {model_type!r}")

    if architecture is None:
        declared = config.architectures
        if not isinstance(declared, (list, tuple)) or len(declared) != 1:
            raise ValueError(
                f"{model_type} recovery requires exactly one architecture in config.architectures "
                "or an explicit --architecture"
            )
        architecture = declared[0]

    # Use the native registry's names so untrusted checkpoint metadata can
    # select a supported task but cannot supply a Python module to load.
    tasks = (
        (auto.AutoModel, auto.MODEL_MAPPING_NAMES),
        (auto.AutoModelForMaskedLM, auto.MODEL_FOR_MASKED_LM_MAPPING_NAMES),
        (
            auto.AutoModelForSequenceClassification,
            auto.MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
        ),
        (auto.AutoModelForTokenClassification, auto.MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES),
        (auto.AutoModelForQuestionAnswering, auto.MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
        (auto.AutoModelForMultipleChoice, auto.MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES),
        (auto.AutoModelForPreTraining, auto.MODEL_FOR_PRETRAINING_MAPPING_NAMES),
        (
            auto.AutoModelForNextSentencePrediction,
            auto.MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
        ),
    )
    for wrapper, names in tasks:
        native = names.get(model_type, ())
        if isinstance(native, str):
            native = (native,)
        if architecture in native:
            return wrapper
    raise ValueError(f"unsupported native {model_type} architecture {architecture!r}")
