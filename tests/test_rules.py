import random

from shareguard.app.synthetic import make_aadhaar, make_card, make_key, make_pan
from shareguard.rules.detectors import detect, has_field_label, luhn_valid, verhoeff_valid


def kinds(s):
    return {f.kind for f in detect(s)}


def test_checksums():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert verhoeff_valid("2363")  # standard Verhoeff example 236 -> check 3
    assert not verhoeff_valid("2364")


def test_generated_secrets_detected():
    r = random.Random(0)
    for _ in range(300):
        assert "aadhaar" in kinds(make_aadhaar(r))
        assert "card" in kinds(make_card(r))
        assert "pan" in kinds(make_pan(r))
        assert "aws_key" in kinds(make_key(r))


def test_confusable_ocr_swaps_still_caught():
    r = random.Random(3)
    a = make_aadhaar(r)
    swapped = a.replace("0", "O").replace("1", "l")
    assert "aadhaar" in kinds(swapped)


def test_safe_text_not_flagged():
    for s in ["Agenda: review latency charts", "The quick brown fox jumps over the lazy dog",
              "Meeting at 10:30 in room 4B", "Version 2.3.1 released"]:
        assert not detect(s), s


def test_field_labels():
    assert has_field_label("Name:")
    assert has_field_label("AWS_ACCESS_KEY_ID=AKIA12")
    assert not has_field_label("Agenda for today")


def test_ocr_spaced_key_label_and_partial_key():
    assert has_field_label("AWS ACCESS KEY ID=AKIA29326ML6")
    assert "aws_key_partial" in kinds("AWS ACCESS KEY ID=AKIA29326ML6")
