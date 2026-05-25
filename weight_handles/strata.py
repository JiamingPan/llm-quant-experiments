"""
Public prompt strata for feature-handle evaluation.
"""


DEFAULT_STRATA = {
    "prose": [
        "The city archive kept letters from travelers who crossed the mountains during winter.",
        "A researcher compared two explanations and wrote a careful note about where they diverged.",
    ],
    "code": [
        "def normalize(xs):\n    total = sum(xs)\n    return [x / total for x in xs if total != 0]",
        "class Cache:\n    def __init__(self):\n        self.store = {}\n    def get(self, key):\n        return self.store.get(key)",
    ],
    "math": [
        "If a sequence satisfies a_n = 2a_{n-1} + 1 and a_0 = 0, compute the first four terms.",
        "Let f(x) = x^2 - 3x + 2. Explain how to find its roots.",
    ],
    "chat": [
        "User: Can you summarize the meeting notes?\nAssistant:",
        "User: I need a concise plan for tomorrow.\nAssistant:",
    ],
    "long_context": [
        "Paragraph one describes an old navigation method. "
        "Paragraph two introduces a conflicting account from another witness. "
        "Paragraph three asks which source is more reliable and why. " * 24,
    ],
    "separators_bos": [
        "<s>\n\n### Instruction:\nExplain the difference between a variable and a parameter.\n\n### Response:",
        ":::: SECTION A ::::\nkey=value\n:::: SECTION B ::::\nWhat changed between the sections?",
    ],
    "sentinel": [
        "The sentinel token is [ANCHOR]. Repeat the word after [ANCHOR] only if it appears later.",
        "BEGIN_RECORD\nfield: alpha\nfield: beta\nEND_RECORD\nWhich field came first?",
    ],
}


def load_strata(selected=None):
    """
    Return a deterministic mapping from stratum name to prompt list.
    """
    if selected is None:
        return {key: list(value) for key, value in DEFAULT_STRATA.items()}
    selected = set(selected)
    return {key: list(value) for key, value in DEFAULT_STRATA.items() if key in selected}


def flatten_strata(strata):
    texts = []
    for name in sorted(strata):
        texts.extend(strata[name])
    return texts
