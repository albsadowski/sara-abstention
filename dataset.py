#!/usr/bin/env python3

import warnings

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module="pydantic",
)

import random
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from sara_abstention.llm import chat_model


random.seed(42)


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-5-nano")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--n", type=int, default=None, help="Limit source records (for debugging)"
    )
    return p.parse_args()


def load_data(task: str, filename: str):
    return pd.read_csv(
        Path("./legalbench/data") / task / filename,
        sep="\t",
        index_col="index",
    )


class KeyFact(BaseModel):
    """
    A single fact extracted from a case description.

    """

    fact_id: str = Field(
        description="Short snake_case id, e.g. 'gross_income_alice_2018'"
    )
    role: Literal["numeric_input", "categorical"] = Field(
        description=(
            "'numeric_input' if this is a specific dollar amount, wage, or "
            "deduction figure that is arithmetically required to compute the "
            "answer to the tax question. A fact is only numeric_input if "
            "removing it would make it impossible to calculate the tax owed. "
            "'categorical' if this is a qualitative status or choice such as "
            "filing status, deduction type, marital status, or dependent "
            "eligibility."
        )
    )
    category: Literal[
        "income",
        "deduction_amount",
        "wages_paid",
        "filing_status",
        "deduction_type",
        "marital_status",
        "dependent",
        "residence",
        "employment",
        "date",
        "other",
    ]
    summary: str = Field(
        description="Human-readable summary, e.g. 'Alice gross income 2018 = $324311'"
    )
    verbatim_span: str = Field(
        description=(
            "The shortest contiguous substring of the original text that "
            "expresses this fact — copy-paste exactly, preserving punctuation "
            "and numbers."
        )
    )
    is_zero_value: bool = Field(
        description=(
            "True if this fact states a zero or null quantity, e.g. "
            "'Bob had no income in 2017'. False otherwise."
        )
    )


class ExtractedFacts(BaseModel):
    facts: list[KeyFact]


EXTRACT_PROMPT = """\
You are a tax-law analyst. Given the case description and question below, \
extract every fact from the description that is relevant to answering the \
tax question.

For each fact, classify its **role**:
- "numeric_input": a specific dollar amount, wage figure, or deduction amount \
  that is **arithmetically required** to compute the tax for the year asked \
  about in the question. Only classify a fact as numeric_input if removing it \
  would make the tax calculation impossible. Be careful with multi-year data: \
  if the question asks about 2018 tax, a 2017 income figure is only \
  numeric_input if the tax statute for 2018 actually requires the 2017 value \
  as input. If in doubt, classify as "categorical".
- "categorical": a qualitative status or choice — filing status \
  (joint / separate / single / head-of-household), deduction type \
  (standard / itemized), marital status, dependent eligibility, residence, \
  employment relationship, dates.

Also flag whether the fact states a **zero or null quantity** (is_zero_value).

For each fact, copy the shortest exact substring from the description that \
states this fact (verbatim_span). This must be a direct copy-paste.

Description:
{description}

Question:
{question}
"""


def extract_facts(llm, description: str, question: str) -> list[KeyFact]:
    result = llm.with_structured_output(ExtractedFacts).invoke(
        EXTRACT_PROMPT.format(description=description, question=question)
    )
    return result.facts


class RephrasedDescription(BaseModel):
    rephrased: str


REDACT_PROMPT = """\
You are a legal text editor. Below is a case description and one specific \
fact that must be removed.

Rewrite the full description so that the fact is *naturally absent*:
- If the fact is its own sentence, drop the sentence.
- If the fact is part of a larger sentence, rephrase or restructure so the \
  specific numeric value is gone, but surrounding context is preserved where \
  possible.  For example, "Alice was paid $73200 in 2015 as an employee of \
  Bertha's Mussels" → "Alice worked in 2015 as an employee of Bertha's \
  Mussels" (the employment context is kept; the dollar figure is gone).
- Do NOT leave placeholders like [REDACTED], "___", or ellipses.
- Do NOT add any sentences, commentary, or the question itself.
- Keep every other fact exactly as-is (same numbers, dates, names, wording).

Return only the rewritten description.

Original description:
{description}

Fact to remove:
{fact_summary}

Verbatim span to eliminate:
{span}
"""


def generate_redacted(llm, description: str, fact: KeyFact) -> str:
    result = llm.with_structured_output(RephrasedDescription).invoke(
        REDACT_PROMPT.format(
            description=description,
            fact_summary=fact.summary,
            span=fact.verbatim_span,
        )
    )
    return result.rephrased


class ContradictedDescription(BaseModel):
    rewritten: str


def _extract_taxpayer_and_year(question: str) -> tuple[str | None, str | None]:
    """
    Pull the taxpayer name and year from the question.

    """
    m = re.search(r"How much tax does (\w+) have to pay in (\d{4})", question)
    if m:
        return m.group(1), m.group(2)
    return None, None


def _extract_names_from_span(span: str) -> list[str]:
    """
    Pull capitalised names from a span (heuristic).

    """
    candidates = re.findall(r"\b([A-Z][a-z]+)\b", span)
    stopwords = {
        "In",
        "On",
        "The",
        "From",
        "For",
        "And",
        "Or",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "January",
        "February",
        "March",
        "April",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "USA",
        "Maryland",
        "Baltimore",
        "Take",
        "Took",
        "Alice",
        "Bob",
        "Charlie",
        "Dorothy",
    }
    names = []
    for c in candidates:
        if c not in stopwords or c in {"Alice", "Bob", "Charlie", "Dorothy"}:
            names.append(c)
    return names


def generate_contrary_assertion(fact: KeyFact, question: str) -> str | None:
    """
    Return a full, grammatically correct sentence that contradicts the given
    categorical fact, or None if we don't have a mapping for it.

    """
    span_lower = fact.verbatim_span.lower()
    taxpayer, year = _extract_taxpayer_and_year(question)
    names = _extract_names_from_span(fact.verbatim_span)

    if fact.category == "filing_status":
        filers = [n for n in names if n in {"Alice", "Bob", "Charlie", "Dorothy"}]
        filer_str = " and ".join(filers) if filers else taxpayer or "They"
        yr = year or "the tax year"

        if "jointly" in span_lower or "joint" in span_lower:
            return f"{filer_str} file separately in {yr}."
        if "separately" in span_lower or "separate" in span_lower:
            return f"{filer_str} file jointly in {yr}."
        if "single" in span_lower:
            return f"{filer_str} file jointly in {yr}."
        if "head of household" in span_lower:
            return f"{filer_str} file as single in {yr}."
        return None

    if fact.category == "deduction_type":
        filers = [n for n in names if n in {"Alice", "Bob", "Charlie", "Dorothy"}]
        filer_str = " and ".join(filers) if filers else taxpayer or "They"
        yr = year or "the tax year"

        if "standard deduction" in span_lower:
            return f"In {yr}, {filer_str} takes itemized deductions."
        if "itemized" in span_lower:
            return f"In {yr}, {filer_str} takes the standard deduction."
        return None

    if fact.category == "marital_status":
        who = taxpayer or "Alice"
        yr = year or "the tax year"

        if "married" in span_lower or "got married" in span_lower:
            return f"{who} was not married during {yr}."
        if "divorced" in span_lower or "separated" in span_lower:
            return (
                f"{who} and her spouse were still legally married "
                f"and living together during {yr}."
            )
        return None

    if fact.category == "dependent":
        who = taxpayer or "Alice"
        yr = year or "the tax year"

        if any(kw in span_lower for kw in ["son", "daughter", "child"]):
            return f"{who} had no dependents during {yr}."
        return None

    return None


CONTRADICT_PROMPT = """\
You are a legal text editor. Below is a case description that contains a \
specific fact. You must add a **contradictory assertion** so the description \
contains two logically incompatible claims.

Rules:
- Insert the contradictory sentence provided below verbatim into the \
  description. Place it at least one sentence away from the original fact.
- Do NOT use hedging language: no "however", "alternatively", "another record \
  states", "note:", "in contrast", or anything similar. The contradictory \
  sentence is already written for you — just insert it.
- Do NOT remove or change the original fact — both the original and the \
  contradictory claim must be present.
- Do NOT add commentary, questions, or the question text itself.
- Keep all other facts exactly as-is.

Original description:
{description}

Original fact (keep this as-is):
{original_span}

Contradictory sentence to insert (use exactly as written):
{contrary_assertion}
"""


def generate_contradicted(
    llm, description: str, fact: KeyFact, contrary_assertion: str
) -> str:
    result = llm.with_structured_output(ContradictedDescription).invoke(
        CONTRADICT_PROMPT.format(
            description=description,
            original_span=fact.verbatim_span,
            contrary_assertion=contrary_assertion,
        )
    )
    return result.rewritten


def _extract_dollar_amount(span: str) -> str | None:
    """
    Extract the specific dollar amount being targeted from a fact summary.

    """
    m = re.search(r"\$[\d,]+", span)
    return m.group() if m else None


def validate_redaction(original: str, redacted: str, fact: KeyFact) -> list[str]:
    issues = []
    target_dollar = _extract_dollar_amount(fact.verbatim_span)
    if target_dollar:
        target_raw = target_dollar.replace(",", "")
        if target_raw in redacted or target_dollar in redacted:
            issues.append(f"target value {target_dollar} still present")

    for marker in ["[REDACTED]", "[redacted]", "___", "…", "Question:"]:
        if marker in redacted and marker not in original:
            issues.append(f"leakage/placeholder: {marker!r}")

    if len(redacted) > len(original) + 20:
        issues.append("redacted text is longer than original")

    return issues


def validate_contradiction(
    original: str, contradicted: str, contrary_assertion: str
) -> list[str]:
    issues = []

    for marker in [
        "however",
        "another record",
        "note:",
        "alternatively",
        "in contrast",
    ]:
        if marker in contradicted.lower() and marker not in original.lower():
            issues.append(f"hedging language: {marker!r}")

    for marker in ["Question:", "\nQuestion"]:
        if marker in contradicted and marker not in original:
            issues.append(f"prompt leakage: {marker!r}")

    contrary_normalised = " ".join(contrary_assertion.split())
    contradicted_normalised = " ".join(contradicted.split())
    if contrary_normalised not in contradicted_normalised:
        issues.append("contrary assertion not found verbatim in output")

    return issues


def main():
    args = parse_args()
    load_dotenv()

    data = load_data("sara_numeric", "test.tsv")
    if args.n is not None:
        data = data.head(args.n)

    llm = chat_model(args.model, args.no_cache)

    rows = []

    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Cases"):
        base = row.drop("text").to_dict()
        case_id = base.get("case id", idx)

        rows.append(
            {
                **base,
                "perturbation": "none",
                "perturbed_fact_id": None,
                "perturbed_fact_role": None,
                "perturbed_category": None,
                "perturbation_detail": None,
                "expected_behaviour": "answer",
                "validation_warnings": None,
            }
        )

        try:
            facts = extract_facts(llm, row["description"], row["question"])
        except Exception as e:
            print(f"[{case_id}] fact extraction failed: {e}")
            continue

        numeric_facts = [f for f in facts if f.role == "numeric_input"]
        categorical_facts = [f for f in facts if f.role == "categorical"]

        redactable_facts = [f for f in numeric_facts if not f.is_zero_value]
        skipped_zero = len(numeric_facts) - len(redactable_facts)

        print(
            f"[{case_id}] extracted {len(numeric_facts)} numeric "
            f"({skipped_zero} zero-value skipped), "
            f"{len(categorical_facts)} categorical facts"
        )

        for fact in redactable_facts:
            try:
                redacted = generate_redacted(llm, row["description"], fact)
                warns = validate_redaction(row["description"], redacted, fact)
                rows.append(
                    {
                        **base,
                        "description": redacted,
                        "perturbation": "redact",
                        "perturbed_fact_id": fact.fact_id,
                        "perturbed_fact_role": fact.role,
                        "perturbed_category": fact.category,
                        "perturbation_detail": f"removed: {fact.summary}",
                        "expected_behaviour": "refuse",
                        "validation_warnings": ("; ".join(warns) if warns else None),
                    }
                )
            except Exception as e:
                print(f"[{case_id}] redact failed for {fact.fact_id}: {e}")

        for fact in categorical_facts:
            contrary = generate_contrary_assertion(fact, row["question"])
            if contrary is None:
                continue

            try:
                contradicted = generate_contradicted(
                    llm, row["description"], fact, contrary
                )
                warns = validate_contradiction(
                    row["description"], contradicted, contrary
                )
                rows.append(
                    {
                        **base,
                        "description": contradicted,
                        "perturbation": "contradict",
                        "perturbed_fact_id": fact.fact_id,
                        "perturbed_fact_role": fact.role,
                        "perturbed_category": fact.category,
                        "perturbation_detail": (
                            f"original: {fact.verbatim_span} | injected: {contrary}"
                        ),
                        "expected_behaviour": "flag_ambiguity",
                        "validation_warnings": ("; ".join(warns) if warns else None),
                    }
                )
            except Exception as e:
                print(f"[{case_id}] contradict failed for {fact.fact_id}: {e}")

    output_df = pd.DataFrame(rows)

    print(f"\n{'=' * 60}")
    print(f"Total rows: {len(output_df)}")
    print(f"\nBy perturbation:")
    print(output_df["perturbation"].value_counts().to_string())
    print(f"\nRedactions by category:")
    redacts = output_df[output_df["perturbation"] == "redact"]
    if len(redacts):
        print(redacts["perturbed_category"].value_counts().to_string())
    print(f"\nContradictions by category:")
    contras = output_df[output_df["perturbation"] == "contradict"]
    if len(contras):
        print(contras["perturbed_category"].value_counts().to_string())

    n_warned = output_df["validation_warnings"].notna().sum()
    print(f"\nValidation warnings: {n_warned} rows flagged")
    if n_warned:
        flagged = output_df[output_df["validation_warnings"].notna()]
        for _, r in flagged.iterrows():
            print(
                f"  [{r.get('case id', '?')}] {r['perturbation']} "
                f"{r['perturbed_fact_id']}: {r['validation_warnings']}"
            )

    output_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(output_df)} rows → {args.output}")


if __name__ == "__main__":
    main()
