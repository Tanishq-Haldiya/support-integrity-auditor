import pytest

from sia.rubric import CRITICAL, HIGH, LOW, MEDIUM, assess, find_cues


@pytest.mark.parametrize("text, level", [
    ("Hi Support, I think my account has been compromised.", CRITICAL),
    ("Hi Support, Lock my account immediately, I see transactions I didn't make.", CRITICAL),
    ("Hi Support, Someone used my card to make a purchase on your site.", CRITICAL),
    ("Hi Support, I received a login alert from a device in Russia, which wasn't me.", CRITICAL),
    ("Hi Support, The application crashes every time I open the settings tab.", HIGH),
    ("Hi Support, We are receiving a 500 Internal Server Error on the API endpoint.", HIGH),
    ("Hi Support, My data hasn’t synced to the cloud for 24 hours.", HIGH),
    ("Hi Support, I lost my phone and cannot pass the 2FA check.", HIGH),
    ("Hi Support, I cannot log into my account even after resetting the password.", HIGH),
    ("Hi Support, I noticed a double charge on my statement for last month.", MEDIUM),
    ("Hi Support, I am not receiving the password reset email.", MEDIUM),
    ("Hi Support, How do I install the latest patch on Windows 11?", LOW),
    ("Hi Support, Where is your headquarters located?", LOW),
])
def test_template_levels(text, level):
    assert assess(text).level == level


TEMPLATE_LEVELS = {
    CRITICAL: ["Why is there a new device added to my trusted list?",
               "Someone used my card to make a purchase on your site.",
               "I received a login alert from a device in Russia, which wasn't me.",
               "I received a suspicious email claiming to be support.",
               "Lock my account immediately, I see transactions I didn't make.",
               "I think my account has been compromised."],
    HIGH: ["We are receiving a 500 Internal Server Error on the API endpoint.",
           "The dashboard is not loading any data, just a spinning wheel.",
           "The application crashes every time I open the settings tab.",
           "I lost my phone and cannot pass the 2FA check.",
           "My data hasn't synced to the cloud for 24 hours.",
           "I cannot log into my account even after resetting the password."],
    MEDIUM: ["Why is my bill higher than the agreed amount?",
             "I requested a refund 5 days ago, where is it?",
             "I noticed a double charge on my statement for last month.",
             "My profile picture is not updating.",
             "My subscription renewed automatically, but I wanted to cancel.",
             "I have been trying to update my payment method but it keeps failing.",
             "I am not receiving the password reset email."],
    LOW: ["Where is your headquarters located?", "What are your support operating hours?",
          "Please help me change the email address associated with my account.",
          "Is there a roadmap for new features this year?",
          "Can you send me the invoice for transaction #99283?",
          "How does the team pricing tier work?", "How do I upgrade to the Enterprise plan?",
          "How do I install the latest patch on Windows 11?", "Do you offer a discount for non-profits?",
          "I would like to request a demo for my team.", "I want to permanently delete my account and data."],
}


@pytest.mark.parametrize("level, text", [(lvl, t) for lvl, ts in TEMPLATE_LEVELS.items() for t in ts])
def test_every_dataset_template_gets_its_intended_level(level, text):
    assert assess("Hi Support, " + text).level == level


def test_filler_words_do_not_trigger_cues():
    # "down", "manager", "card", "security" and "stop" appear in the dataset's random filler text.
    text = "Hi Support, Where is your headquarters located? Down manager card security stop break."
    assert assess(text).level == LOW
    # Real filler sentence from the dataset that used to match the missing-email cue.
    assert assess("Hi Support, Is there a roadmap for new features this year? Spring never arrive shoulder.").level == LOW


def test_urgency_raises_routine_request_one_level():
    assert assess("URGENT: please send the invoice today.").level == MEDIUM
    assert assess("No rush, but please send the invoice.").level == LOW


def test_urgency_does_not_push_high_to_critical():
    assert assess("Urgent: the app crashes on launch.").level == HIGH


def test_subject_only_used_when_description_is_too_short():
    assert assess("Hi Support, help!", subject="Account hacked").level == CRITICAL
    assert assess("Hi Support, Where is your headquarters located?", subject="Account hacked").level == LOW


def test_match_spans_are_exact_substrings():
    text = "Hi Support, The dashboard is not loading any data, just a spinning wheel."
    for m in find_cues(text):
        assert text[m.start:m.end] == m.text
