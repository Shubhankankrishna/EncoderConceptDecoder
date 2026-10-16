"""Normalize the upstream parquet's serialized Python byte literals safely."""
import ast


def clean_math_field(value):
    if isinstance(value, bytes):
        value=value.decode('utf-8',errors='strict')
    elif not isinstance(value,str):
        raise TypeError('Math fields must be UTF-8 bytes or strings')
    stripped=value.strip()
    if stripped.startswith(("b'",'b"')):
        # literal_eval parses a literal only; never executes dataset code.
        parsed=ast.literal_eval(stripped)
        if not isinstance(parsed,bytes):raise ValueError('Expected a bytes literal')
        value=parsed.decode('utf-8',errors='strict')
    value=value.strip()
    if not value:raise ValueError('Empty math field')
    return value
