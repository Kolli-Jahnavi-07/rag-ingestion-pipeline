from ragpipe.chunking import chunk_document
from ragpipe.text import boilerplate, redact, sections, strip_boilerplate, tokenize

DOC = """# Late Payments in Ohio

Intro line.

## Grace period

A payment is late after 10 days. No charge in the grace period.

## Late charge

The charge is the lesser of $25 or 5% of the installment.
"""


def test_redaction_masks_pii_before_indexing():
    text, n = redact("SSN 123-45-6789, acct 4111111111111111, call (614) 555-0100 or jane.doe@example.com")
    assert n == 4
    assert "123-45-6789" not in text and "4111111111111111" not in text and "555-0100" not in text and "@" not in text


def test_sections_keep_the_heading_trail():
    title, secs = sections(DOC)
    assert title == "Late Payments in Ohio"
    assert [s.path for s in secs] == [
        ("Late Payments in Ohio",),
        ("Late Payments in Ohio", "Grace period"),
        ("Late Payments in Ohio", "Late charge"),
    ]


def test_boilerplate_is_detected_across_documents_and_removed():
    footer = "This document is for internal use."
    docs = [f"# A\n\nbody {i}\n\n{footer}" for i in range(4)] + ["# B\n\nother"]
    common = boilerplate(docs)
    assert common == {footer}
    assert footer not in strip_boilerplate(docs[0], common)


def test_tokenizer_keeps_money_and_percent_drops_stopwords():
    assert tokenize("The fee is $25 or 5% of the installment") == ["fee", "$25", "5%", "installment"]


def test_chunks_carry_context_and_have_stable_ids():
    title, secs = sections(DOC)
    a = chunk_document("policies/oh/late_payments", title, secs)
    b = chunk_document("policies/oh/late_payments", title, secs)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    late = next(c for c in a if c.heading.endswith("Late charge"))
    assert late.text.startswith("Late Payments in Ohio > Late charge\n")


def test_long_sections_split_with_overlap():
    body = " ".join(f"Sentence number {i} has exactly seven words." for i in range(60))
    _, secs = sections(f"# T\n\n## S\n\n{body}")
    chunks = chunk_document("d", "T", secs, max_words=60, overlap_words=15)
    assert len(chunks) > 5
    assert all(c.words <= 60 + 8 for c in chunks)  # body limit plus the header line
    for a, b in zip(chunks, chunks[1:], strict=False):
        last_sentence = a.text.split(". ")[-1].rstrip(".")
        assert last_sentence in b.text  # overlap carries the tail forward
