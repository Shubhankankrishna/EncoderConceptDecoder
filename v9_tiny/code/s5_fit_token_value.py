"""Fit the token tree's value head on S4's executed trajectories with the chosen token decoder (§9.6)."""
import argparse
import torch
import v9paths
import s5_run

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--s4', required=True); ap.add_argument('--out', required=True); ap.add_argument('--token-adapter')
    a = ap.parse_args(); torch.set_num_threads(4)
    s5_run.fit_token_value(s5_run.token_model(a.model, a.token_adapter), a.s4, a.out)
