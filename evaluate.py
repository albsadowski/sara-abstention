#!/usr/bin/env python3

import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional


warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module="pydantic",
)


from datetime import date

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from sara_abstention.llm import chat_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    if args.output is None:
        stamp = date.today().strftime("%Y%m%d")
        args.output = f"./results/{args.model}_{stamp}.csv"
    return args


class TaxLiability(BaseModel):
    monetary_result: Optional[float] = Field(
        None,
        description=(
            "The computed monetary result in USD. "
            "Null if the answer cannot be determined from the provided facts."
        ),
    )


EVAL_PROMPT = """\
You are an expert tax attorney. Given the applicable statute, the taxpayer's facts, \
and the question below, determine the correct monetary answer.

Statute:
{statute}

Facts:
{description}

Question:
{question}
"""


def query_llm(llm, statute: str, description: str, question: str) -> Optional[float]:
    result = llm.with_structured_output(TaxLiability).invoke(
        EVAL_PROMPT.format(statute=statute, description=description, question=question)
    )
    return result.monetary_result


def main():
    args = parse_args()
    load_dotenv()

    data = pd.read_csv(args.input)
    llm = chat_model(args.model, args.no_cache)

    rows = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        try:
            llm_answer = query_llm(
                llm, row["statute"], row["description"], row["question"]
            )
        except Exception as e:
            print(f"Warning: query failed for case {row.get('case id', '?')}: {e}")
            llm_answer = None

        rows.append({**row.to_dict(), "model": args.model, "llm_answer": llm_answer})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
