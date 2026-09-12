from .memory import SemanticMemory, SemanticTripleEntry


def __getattr__(name):
    if name == "SemanticExtraction":
        from .semantic_extraction import SemanticExtraction
        return SemanticExtraction
    if name == "SemanticConsolidation":
        from .semantic_consolidation import SemanticConsolidation
        return SemanticConsolidation
    raise AttributeError(name)
