"""Reuse the immutable, cleaned V10.1 corpus and tokenizer; never rebuild in place."""
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]/"v10_1/data"
