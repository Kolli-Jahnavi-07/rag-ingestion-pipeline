"""A synthetic knowledge base for a loan-servicing support assistant, plus the questions to grade it.

Policy documents exist per state and topic, all written from the same templates, so documents
are near-duplicates of each other except for the state and the numbers. That's what makes
retrieval hard in real policy libraries, and it's what the evaluation measures.

Version 2 of the corpus simulates the next week's sync: some policies change their numbers,
some programs are retired (documents deleted), new ones are added, and an FAQ is copied under a
second filename. All content is fictional.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

STATES = {
    "OH": "Ohio",
    "TX": "Texas",
    "CA": "California",
    "FL": "Florida",
    "GA": "Georgia",
    "MI": "Michigan",
    "NC": "North Carolina",
    "AZ": "Arizona",
    "IL": "Illinois",
    "PA": "Pennsylvania",
    "NY": "New York",
    "MA": "Massachusetts",
}

FOOTER = (
    "\n\n---\nThis document is for internal servicing use. Contact the compliance team with questions. "
    "Customer communications must follow approved templates.\n"
)


@dataclass(frozen=True)
class Facts:
    grace_days: int
    late_fee: int
    late_fee_pct: int
    repo_notice_days: int
    payoff_valid_days: int
    title_days: int
    hardship_months: int
    mileage_fee_cents: int
    inspection_days: int


@dataclass(frozen=True)
class Question:
    text: str
    doc_id: str
    answer: str  # the fact a correct retrieval must contain
    state: str | None


def facts_for(state: str, rng: random.Random) -> Facts:
    return Facts(
        grace_days=rng.choice([10, 10, 15, 15, 20]),
        late_fee=rng.choice([15, 20, 25, 30, 35, 40]),
        late_fee_pct=rng.choice([4, 5, 5, 6, 8]),
        repo_notice_days=rng.choice([10, 15, 20, 21, 30]),
        payoff_valid_days=rng.choice([7, 10, 10, 14]),
        title_days=rng.choice([10, 15, 20, 30]),
        hardship_months=rng.choice([2, 3, 3, 4, 6]),
        mileage_fee_cents=rng.choice([15, 20, 25, 30]),
        inspection_days=rng.choice([30, 45, 60, 90]),
    )


def _late(st: str, f: Facts) -> str:
    n = STATES[st]
    return f"""# Late Payments in {n}

Applies to retail installment contracts for customers whose billing address is in {n} ({st}).

## Grace period

A payment is considered late when it is not received within {f.grace_days} days after the due date.
No late charge may be assessed during the grace period.

## Late charge

After the grace period, the late charge in {n} is the lesser of ${f.late_fee} or {f.late_fee_pct}% of the
unpaid installment. Only one late charge may be assessed per installment.

## Account notes

Record the date the payment was received and whether the late charge was assessed or waived.
Waivers above two per year require supervisor approval.
"""


def _repo(st: str, f: Facts) -> str:
    n = STATES[st]
    return f"""# Repossession Notices in {n}

## Notice of default

Before a vehicle can be assigned for repossession in {n}, the customer must receive a written notice of
default and right to cure at least {f.repo_notice_days} days in advance.

## After repossession

Personal property found in the vehicle is inventoried and held for the customer. The post-repossession
notice must state how the customer can redeem the vehicle and the amount required.
"""


def _payoff(st: str, f: Facts) -> str:
    n = STATES[st]
    return f"""# Payoff Quotes and Title Release in {n}

## Payoff quote

A payoff quote for a {n} account is valid for {f.payoff_valid_days} days from the date it is issued. Quotes
include per diem interest through the expiration date.

## Title release

After the final payment clears, the lien release or title is sent within {f.title_days} days. Electronic
titles are released through the state system on the same schedule.
"""


def _hardship(st: str, f: Facts) -> str:
    n = STATES[st]
    return f"""# Hardship Assistance in {n}

## Payment deferral

Customers in {n} experiencing a documented hardship may defer up to {f.hardship_months} monthly payments over the
life of the contract. Deferred payments are added to the end of the term.

## Eligibility

The account must be no more than 59 days past due at the time of the request, and at least six payments
must have been made since origination.
"""


def _lease_end(st: str, f: Facts) -> str:
    n = STATES[st]
    return f"""# Lease-End Return in {n}

## Inspection

Lessees in {n} can schedule a free pre-return inspection up to {f.inspection_days} days before the lease ends.

## Excess mileage

Miles above the contracted allowance are charged at {f.mileage_fee_cents} cents per mile for leases written in {n}.
Wear and use charges follow the standard guide.
"""


TOPICS = {
    "late_payments": _late,
    "repossession_notices": _repo,
    "payoff_and_title": _payoff,
    "hardship_assistance": _hardship,
    "lease_end": _lease_end,
}

FAQS = {
    "faq/autopay": """# Autopay

## Enrolling

Customers can enroll in autopay from the online account or the mobile app. The first automatic payment is
drafted on the next due date at least five business days after enrollment.

## Changing the draft date

The draft date can be moved once every twelve months, and only within the same calendar month.
""",
    "faq/paperless": """# Paperless Statements

## Turning on paperless

Paperless statements are enabled in account settings. Customers receive an email when each statement is ready.
Paper statements stop after the next billing cycle.
""",
    "faq/account_security": """# Account Security

## Verifying identity

Agents verify identity with the last four digits of the account number and the date of birth before discussing
an account. Never ask for a full Social Security number by phone. Example in notes: SSN 123-45-6789 must be masked.
""",
}


def corpus(version: int = 1, seed: int = 11) -> tuple[dict[str, str], dict[str, Facts]]:
    """The documents ({doc_id: markdown}) and the facts they state, for a given corpus version."""
    rng = random.Random(seed)
    facts = {st: facts_for(st, rng) for st in STATES}
    docs: dict[str, str] = {}
    for st in STATES:
        for topic, render in TOPICS.items():
            if topic == "lease_end" and st in ("NY", "MA") and version == 1:
                continue  # lease programs in these states launch in version 2
            docs[f"policies/{st.lower()}/{topic}"] = render(st, facts[st]) + FOOTER
    docs.update({k: v + FOOTER for k, v in FAQS.items()})

    if version >= 2:
        upd = random.Random(seed + 1)
        for st in sorted(upd.sample(list(STATES), 5)):  # policy changes
            f = facts[st]
            facts[st] = Facts(**{**f.__dict__, "grace_days": f.grace_days + 5, "late_fee": f.late_fee + 10})
            docs[f"policies/{st.lower()}/late_payments"] = _late(st, facts[st]) + FOOTER
        for st in ("GA", "AZ"):  # programs retired
            docs.pop(f"policies/{st.lower()}/hardship_assistance", None)
        docs["faq/autopay_copy"] = docs["faq/autopay"]  # same file uploaded twice by an editor
    return docs, facts


def build(root: Path, version: int = 1, seed: int = 11) -> dict[str, str]:
    """Write the corpus as Markdown files under `root` (replacing what's there) and return it."""
    docs, _ = corpus(version, seed)
    if root.exists():
        for p in root.rglob("*.md"):
            p.unlink()
    for doc_id, text in docs.items():
        p = root / f"{doc_id}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return docs


def questions(version: int = 1, seed: int = 11) -> list[Question]:
    """Graded questions, phrased the way agents ask them, with the doc and fact that answer each."""
    docs, facts = corpus(version, seed)
    qs: list[Question] = []
    for st, name in STATES.items():
        f = facts[st]
        base = f"policies/{st.lower()}"
        qs += [
            Question(
                f"How long is the grace period before a payment counts as late for a customer in {name}?",
                f"{base}/late_payments",
                f"within {f.grace_days} days",
                st,
            ),
            Question(f"What late charge can we assess on a {name} account?", f"{base}/late_payments", f"${f.late_fee}", st),
            Question(
                f"How many days notice before repossession assignment in {name}?",
                f"{base}/repossession_notices",
                f"{f.repo_notice_days} days in advance",
                st,
            ),
            Question(
                f"How long is a payoff quote good for in {name}?",
                f"{base}/payoff_and_title",
                f"valid for {f.payoff_valid_days} days",
                st,
            ),
            Question(
                f"When does the title get released after payoff for {name} customers?",
                f"{base}/payoff_and_title",
                f"within {f.title_days} days",
                st,
            ),
        ]
        # How agents actually type: abbreviations, shorthand, different words than the policy uses.
        qs += [
            Question(
                f"{st} customer paid a few days after the due date, when does it count as late?",
                f"{base}/late_payments",
                f"within {f.grace_days} days",
                st,
            ),
            Question(f"max late fee for an {st} acct?", f"{base}/late_payments", f"${f.late_fee}", st),
            Question(
                f"{st} repo - how much notice do we owe the customer?",
                f"{base}/repossession_notices",
                f"{f.repo_notice_days} days in advance",
                st,
            ),
            Question(
                f"{st} payoff letter expires after how long?",
                f"{base}/payoff_and_title",
                f"valid for {f.payoff_valid_days} days",
                st,
            ),
            Question(
                f"lien release timing in {st} once the loan is paid off?",
                f"{base}/payoff_and_title",
                f"within {f.title_days} days",
                st,
            ),
        ]
        if f"{base}/hardship_assistance" in docs:
            qs += [
                Question(
                    f"How many payments can a {name} customer defer for hardship?",
                    f"{base}/hardship_assistance",
                    f"up to {f.hardship_months} monthly payments",
                    st,
                ),
                Question(
                    f"how many months can an {st} borrower skip because of a hardship?",
                    f"{base}/hardship_assistance",
                    f"up to {f.hardship_months} monthly payments",
                    st,
                ),
            ]
        if f"{base}/lease_end" in docs:
            qs += [
                Question(
                    f"What is the excess mileage charge on a {name} lease?",
                    f"{base}/lease_end",
                    f"{f.mileage_fee_cents} cents per mile",
                    st,
                ),
                Question(
                    f"{st} lease turn-in: what do we bill per mile over the allowance?",
                    f"{base}/lease_end",
                    f"{f.mileage_fee_cents} cents per mile",
                    st,
                ),
            ]
    qs += [
        Question("How often can a customer change their autopay draft date?", "faq/autopay", "once every twelve months", None),
        Question("When do paper statements stop after going paperless?", "faq/paperless", "after the next billing cycle", None),
        Question("What do agents use to verify identity on a call?", "faq/account_security", "last four digits", None),
    ]
    return qs
