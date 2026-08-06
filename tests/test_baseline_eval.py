"""Part A: the honest evaluation groups boilerplate variants of one message.

The train set wraps a core sentence in greeting/urgency templates
("Urgent: X", "Hey, X. Thanks."). These tests pin the normaliser that the
grouped split relies on: variants must collapse to the same key, distinct
messages must not.
"""
from baseline.baseline_classifier import core


def test_variants_of_same_message_share_a_core():
    a = "Urgent: Where can I download my tax documents for last year?"
    b = "Hello team, Where can I download my tax documents for last year?"
    assert core(a) == core(b)


def test_stacked_prefix_and_suffix_are_both_stripped():
    assert core("Hey, My funds are gone. Please advise.") == "my funds are gone."


def test_distinct_messages_keep_distinct_cores():
    assert core("How do I reset my password?") != core("How do I close my account?")
