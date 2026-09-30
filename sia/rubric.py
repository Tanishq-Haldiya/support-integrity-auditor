"""Rule-based severity rubric.

Each cue is a phrase-level pattern tied to a severity level. Patterns are
deliberately multi-word (e.g. "api ... down", not just "down") so that
unrelated filler text cannot trigger them. Every match keeps its exact span
so that downstream evidence is grounded in the ticket text.
"""

import re
from dataclasses import dataclass, field

from sia import PRIORITY_LEVELS

LOW, MEDIUM, HIGH, CRITICAL = range(4)


@dataclass(frozen=True)
class Cue:
    name: str
    level: int
    pattern: re.Pattern
    description: str


def _cue(name, level, pattern, description):
    return Cue(name, level, re.compile(pattern, re.IGNORECASE), description)


CUES = [
    # --- Critical: security, fraud, account takeover -------------------------
    _cue("account_compromise", CRITICAL,
         r"\b(compromised|hacked|hijacked|taken over|data breach|breached)\b",
         "Customer reports the account is compromised"),
    _cue("unauthorized_transaction", CRITICAL,
         r"\b(unauthori[sz]ed|fraudulent)\b"
         r"|\b(transactions?|purchases?|charges?|payments?)\b[^.!?]*\b(didn't|did not|never|wasn't|was not)\s+(make|made|authori[sz]e|me)\b"
         r"|\bsomeone\s+(else\s+)?(used|is using|has used|charged)\s+my\s+(card|account)\b",
         "Transactions the customer did not make"),
    _cue("stolen_card", CRITICAL,
         r"\bstolen\s+(credit\s+|debit\s+)?card\b|\bcard\s+(was|has been|got)\s+stolen\b",
         "Payment card reported stolen"),
    _cue("phishing", CRITICAL,
         r"\bphishing\b|\bsuspicious\s+(email|message|text|link|sms|call)\b"
         r"|\b(email|message|text|call)\s+(claiming|pretending)\s+to\s+be\b",
         "Possible phishing / impersonation"),
    _cue("unrecognized_access", CRITICAL,
         r"\b(login|log-in|sign-?in)\s+alert\b|\b(new|unknown|unrecognized|strange)\s+device\b"
         r"|\bunrecognized\s+(login|sign-?in|activity)\b|\bwasn't me\b|\bwas not me\b",
         "Account access the customer does not recognise"),

    # --- High: service disruption, blocked access, data integrity ------------
    _cue("crash", HIGH,
         r"\bcrash(es|ed|ing)?\b|\bfreez(es|ing)\b|\bfroze\b",
         "Application crashes or freezes"),
    _cue("server_error", HIGH,
         r"\b5\d\d\b[^.!?]*\berror\b|\berror\s+5\d\d\b|\binternal server error\b|\boutage\b"
         r"|\b(api|endpoint|service|server|site|website|platform)\b[^.!?]*\b(is down|went down|unavailable|timing out|times out)\b",
         "Server errors or service outage"),
    _cue("not_loading", HIGH,
         r"\bnot\s+(loading|load|working|responding|opening)\b|\bwon't\s+(load|open|start|work)\b"
         r"|\bspinning wheel\b|\bblank screen\b",
         "Core functionality is not loading or working"),
    _cue("data_sync_failure", HIGH,
         r"\b(hasn't|has not|haven't|have not|isn't|is not|not|never)\s+(been\s+)?(synced|syncing|sync)\b"
         r"|\bsync(ing)?\s+(failed|failure|error)\b|\bdata loss\b|\blost\s+(my|our|all)\s+(the\s+)?data\b",
         "Data is not syncing or has been lost"),
    _cue("locked_out", HIGH,
         r"\b(cannot|can't|unable to|not able to)\s+(log\s?in(to)?|sign\s?in(to)?|access|get into)\b|\blocked out\b"
         r"|\b(cannot|can't|unable to)\s+(pass|get past|complete)\s+(the\s+)?(2fa|two[- ]factor)\b",
         "Customer is locked out of the product"),

    # --- Medium: billing problems, degraded features, pending requests -------
    _cue("billing_error", MEDIUM,
         r"\bdouble charge|\bcharged twice\b|\bovercharg|\bbill\b[^.!?]*\bhigher than\b"
         r"|\b(wrong|incorrect)\s+(amount|charge|bill)\b",
         "Billing discrepancy"),
    _cue("refund", MEDIUM,
         r"\brefund\b",
         "Outstanding refund"),
    _cue("payment_failure", MEDIUM,
         r"\b(payment|card)\b[^.!?]*\b(keeps? failing|failed|declined|rejected|keeps? getting rejected)\b"
         r"|\bpayment method\b[^.!?]*\b(fail|error)",
         "Payment could not be processed"),
    _cue("unwanted_renewal", MEDIUM,
         r"\brenewed automatically\b|\bauto-?renew|\bwanted to cancel\b",
         "Subscription renewed against the customer's wishes"),
    _cue("missing_email", MEDIUM,
         r"\bnot\s+receiv(ing|ed)\b"
         r"|\b(email|message|notification|link|code|sms)\b[^.!?]*\b(never|didn't|did not|doesn't|does not)\s+(arrives?|arrived|come|came)\b"
         r"|\b(never|didn't|did not)\s+(receive|get|got)\b[^.!?]*\b(email|message|notification|link|code|sms)\b",
         "Expected email/notification never arrived"),
    _cue("feature_defect", MEDIUM,
         r"\bnot\s+(updating|saving|showing)\b|\b(won't|doesn't|does not)\s+(update|save)\b|\bisn't working\b",
         "A feature is not behaving correctly"),

    # --- Low: informational and administrative requests (evidence only) ------
    _cue("how_to_question", LOW,
         r"\bhow\s+(do|can|does|to)\b|\bwhat\s+(is|are)\b|\bwhere\s+(is|are|can)\b|\bis there\b"
         r"|\bdo you (offer|have)\b",
         "Informational / how-to question"),
    _cue("service_request", LOW,
         r"\b(invoice|receipt|pricing|discount|roadmap|demo|operating hours|headquarters|upgrade|enterprise plan)\b"
         r"|\bchange\s+(the\s+|my\s+)?email\b|\bdelete\s+(my\s+)?account\b",
         "Routine administrative request"),
]

# Urgency language raises a Low/Medium assessment by one level.
AMPLIFIER = re.compile(
    r"\b(urgent(ly)?|asap|as soon as possible|emergency|right away|immediately)\b"
    r"|\bspeak\s+(to|with)\s+(a|your)\s+(manager|supervisor)\b|\bunacceptable\b"
    r"|\b(all|every one of)\s+(our\s+)?(users|customers|clients)\b",
    re.IGNORECASE,
)
NEGATED_URGENCY = re.compile(r"\b(not|isn't|no)\s+(urgent|an emergency|rush)\b|\bno rush\b", re.IGNORECASE)

GREETING = re.compile(r"^\s*(hi|hello|hey|dear)\b[^,.!?]{0,40}[,.!]\s*", re.IGNORECASE)
MIN_DESCRIPTION_WORDS = 4


@dataclass(frozen=True)
class CueMatch:
    cue: str
    level: int
    text: str
    start: int
    end: int
    field: str
    description: str

    def to_evidence(self):
        return {
            "signal": "text_cue",
            "cue": self.cue,
            "field": self.field,
            "matched_text": self.text,
            "indicates": PRIORITY_LEVELS[self.level],
            "meaning": self.description,
        }


@dataclass
class Assessment:
    level: int
    matches: list = field(default_factory=list)
    amplified_by: CueMatch = None
    source: str = "description"

    @property
    def severity(self):
        return PRIORITY_LEVELS[self.level]

    @property
    def confidence(self):
        """Heuristic confidence of the rule-based label (used for sample weighting)."""
        supporting = {m.cue for m in self.matches if m.level == self.level}
        if self.level == LOW and not self.matches:
            return 0.5  # nothing matched; defaulted to Low
        return 0.9 if len(supporting) > 1 else 0.75


def _normalise(text):
    # Same-length replacements only, so match offsets stay valid for the original text.
    return str(text).replace("’", "'").replace("‘", "'")


def find_cues(text, field_name="description"):
    if not isinstance(text, str) or not text.strip():
        return []
    norm = _normalise(text)
    matches = []
    for cue in CUES:
        for m in cue.pattern.finditer(norm):
            matches.append(CueMatch(cue.name, cue.level, text[m.start():m.end()],
                                    m.start(), m.end(), field_name, cue.description))
    return sorted(matches, key=lambda m: m.start)


def assess(description, subject=None):
    """Infer a 0-3 severity level from ticket text.

    The description is authoritative. The subject is only consulted when the
    description is too short to judge, since subjects are often auto-generated
    and may not match the body.
    """
    desc = description if isinstance(description, str) else ""
    body = GREETING.sub("", desc)
    matches = find_cues(desc, "description")
    source = "description"
    if len(body.split()) < MIN_DESCRIPTION_WORDS and isinstance(subject, str):
        matches += find_cues(subject, "subject")
        source = "description+subject"

    level = max((m.level for m in matches), default=LOW)

    amplified_by = None
    combined = " ".join(t for t in (desc, subject if source != "description" else "") if t)
    amp = AMPLIFIER.search(_normalise(combined))
    if amp and not NEGATED_URGENCY.search(_normalise(combined)) and level < HIGH:
        level += 1
        amplified_by = CueMatch("urgency_language", level, combined[amp.start():amp.end()],
                                amp.start(), amp.end(), "description",
                                "Customer explicitly signals urgency")

    return Assessment(level=level, matches=matches, amplified_by=amplified_by, source=source)


def evidence(assessment, limit=6):
    """Grounded evidence list: cues at the decisive level first, then the rest."""
    ordered = sorted(assessment.matches, key=lambda m: (-m.level, m.start))
    items = [m.to_evidence() for m in ordered[:limit]]
    if assessment.amplified_by is not None:
        items.append(assessment.amplified_by.to_evidence())
    return items
