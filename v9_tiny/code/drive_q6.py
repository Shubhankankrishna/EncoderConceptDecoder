"""Q6 queue (draft §9.5-§9.6, §14, I17): S5 per promoted seed.
1. Development choices (dev hard10, 256 items, 4x): token decoder (strongest S4 token arm by dev greedy hard10, else the
   promoted decoder), token tree leaf rule (rollout vs value head), and the best inner method for the answer loop.
2. S5_SEALED_PROCEDURE.json is written before any sealed item is read.
3. Primary matrix on sealed hard10 (512) and hard14 (256): RH(1), RH(H*) if H* > 1, G(1), token tree, best-of-N,
   at budgets {1, 4, 16} x t_g. Methods the register disallows are skipped and recorded.
4. Mechanism (H16): grounded RH(1) on sealed hard10 at 16x, then decoder-free RH(1) at grounded RH(1)'s node count.
5. Answer loop (H14): best inner method + AB-MCTS-A, CONT = bounded GradCuit vs random direction, sealed hard10, 16x.
6. Exploratory (first seed only): G(2), G(4), IMAG on 256 dev hard10 at 4x, under the register's conditions."""
import time
import numpy as np
import v9paths
from v9paths import RUNS, SMOKE, CODE, SPEC, read_json, write_json, digest, DATA
from queue_base import Queue

Q6_FILES = [CODE / n for n in ('drive_q6.py', 'queue_base.py', 's5_run.py', 's5_fit_token_value.py', 's5.py', 'csearch.py', 'kernels.py', 'baselines.py', 'train_s4.py', 'hncp.py', 'hgen.py',
                               'measure_s2.py', 'ceiling.py', 'train_s2.py', 'v9paths.py')]
BUDGETS = (1, 4) if SMOKE else (1, 4, 16)
N = dict(hard10=16 if SMOKE else 512, hard14=8 if SMOKE else 256); NDEV = 16 if SMOKE else 256
SEALED = 'dev' if SMOKE else 'v9test'      # smoke runs never read sealed items


class Q6(Queue):
    block = 'Q6'; files = Q6_FILES

    def cell(self, name, args):
        out = RUNS / 'S5' / f'{name}.json'; out.parent.mkdir(exist_ok=True)
        return self.once(name, CODE / 's5_run.py', args + ['--out', out], out)

    def main(self):
        if not SMOKE:
            self.wait_for(RUNS / 'Q5_RESULT.json', 'Q5')
        lock = self.preflight(extra_tests=('q6_smoke.json',)); self.log(stage='Q6', event='preflight_passed', lock=lock['lock_sha256'])
        q5 = read_json(RUNS / 'Q5_RESULT.json'); q3a = read_json(RUNS / 'Q3A_RESULT.json'); cons = q3a['consequences']
        Hs = max(1, cons.get('H_star', 0)); concept = q5['concept_branch']
        rh_ok = bool(cons.get('RH1_allowed')) and concept; g1_ok = concept; imag = cons.get('IMAG', 'not run')
        result = dict(seeds=[], H_star=Hs, skipped=[])
        for si, row in enumerate(q5['seeds']):
            seed = row['seed']; model = row['source']; s4 = str(RUNS / f'S4_s{seed}_EI') if concept else None; tgc = RUNS / 'S5' / f'tg_s{seed}.json'
            base = ['--model', model, '--seed', seed, '--tg-cache', tgc, '--H', Hs] + (['--s4', s4] if s4 else [])
            # 1. Development choices.
            best_tok = None; best = read_json(row['arms']['reference']['greedy'])['hard10']['rate']
            for m in ('rft', 'grpo'):
                if m in row['arms']:
                    r = read_json(row['arms'][m]['greedy'])['hard10']['rate']
                    if r > best:
                        best, best_tok = r, row['arms'][m]['adapter']
            tok = ['--token-adapter', best_tok] if best_tok else []
            dev = lambda meth, extra=(): self.cell(f's{seed}_dev_{meth}', base + tok + ['--method', meth.split('-')[0], '--split', 'dev', '--suite', 'hard10', '--n', NDEV, '--mult', 4] + list(extra))
            tt_choice = 'rollout'; ttv = None
            if s4:
                ttv = RUNS / 'S5' / f'token_value_s{seed}.pt'
                if not ttv.exists():
                    self.run(f's{seed}_token_value', CODE / 's5_fit_token_value.py', ['--model', model, '--s4', s4, '--out', ttv] + tok)
                a_ = dev('token_tree-rollout', ['--tt-leaf', 'rollout']); b_ = dev('token_tree-value', ['--tt-leaf', 'value', '--tt-value', ttv])
                tt_choice = 'value' if b_['rate'] > a_['rate'] else 'rollout'
            tt_args = ['--tt-leaf', tt_choice] + (['--tt-value', ttv] if tt_choice == 'value' else [])
            methods = []
            if rh_ok:
                methods.append(('RH1', ['--method', 'rh', '--H', 1]))
                if Hs > 1:
                    methods.append((f'RH{Hs}', ['--method', 'rh', '--H', Hs]))
            else:
                result['skipped'].append(dict(seed=seed, method='RH', reason='RH(1) not allowed by the register or concept branch stopped'))
            if g1_ok:
                methods.append(('G1', ['--method', 'g', '--R', '1']))
            methods += [('token_tree', ['--method', 'token_tree'] + tt_args), ('best_of_n', ['--method', 'best_of_n'])]
            dev_rates = {}
            for m, args in methods:
                if m in ('RH1', f'RH{Hs}', 'G1'):
                    dev_rates[m] = self.cell(f's{seed}_dev_{m}', base + tok + args + ['--split', 'dev', '--suite', 'hard10', '--n', NDEV, '--mult', 4])['rate']
            best_inner = max(dev_rates, key=dev_rates.get) if dev_rates else None
            # 2. Sealed procedure, fixed before any sealed read.
            proc = (RUNS if SMOKE else SPEC) / f'S5_SEALED_PROCEDURE_s{seed}.json'
            if not proc.exists():
                write_json(proc, dict(seed=seed, model=model, model_sha256=digest(model), s4=s4, token_adapter=best_tok, token_tree_leaf=tt_choice, methods=[m for m, _ in methods],
                                      budgets=list(BUDGETS), suites=N, mechanism=bool(rh_ok), answer_loop_inner=best_inner, dev_rates=dev_rates,
                                      sealed=dict(hard10=digest(DATA / 'v9test_hard10.npz'), hard14=digest(DATA / 'v9test_hard14.npz')), created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
            cells = {}
            # 3. Primary matrix.
            for suite in ('hard10', 'hard14'):
                for b in BUDGETS:
                    for m, args in methods:
                        extra = tok if m in ('token_tree', 'best_of_n') else []
                        cells[f'{m}/{suite}/{b}'] = self.cell(f's{seed}_sealed_{m}_{suite}_{b}x', base + extra + args + ['--split', SEALED, '--suite', suite, '--n', N[suite], '--mult', b])['rate']
            big = BUDGETS[-1]
            # 4. Mechanism comparison (H16).
            if rh_ok:
                g = self.cell(f's{seed}_sealed_RH1grounded_hard10_{big}x', base + ['--method', 'rh_grounded', '--split', SEALED, '--suite', 'hard10', '--n', N['hard10'], '--mult', big])
                steps = max(1, sum(r.get('nodes', 0) for r in g['records']) / max(1, sum(1 for r in g['records'])) / max(1., read_json(tgc)[f'{SEALED}_hard10_{N["hard10"]}']['mean_hops']))
                self.cell(f's{seed}_sealed_RH1nodematched_hard10_{big}x', base + ['--method', 'rh', '--H', 1, '--split', SEALED, '--suite', 'hard10', '--n', N['hard10'], '--mult', big, '--node-match', steps])
            # 5. Answer loop (H14), around the best concept inner method.
            if best_inner and concept:
                gp = read_json(row['gradcuit_pilot']) if row.get('gradcuit_pilot') else dict(chosen={'0.5': 1e-2})
                lr = float(gp['chosen'].get('0.5', 1e-2))
                for kern in ('gradcuit', 'random'):
                    self.cell(f's{seed}_sealed_answerloop_{kern}_hard10_{big}x', base + ['--method', 'answer_loop', '--kernel', kern, '--lr', lr, '--split', SEALED, '--suite', 'hard10', '--n', N['hard10'], '--mult', big])
            # 6. Exploratory block, first seed only.
            if si == 0 and concept:
                comb = read_json(RUNS / 'S2_REGISTER_dense.json')['combined']
                for R in ('2', '4'):
                    if comb.get(f'G2.3_H{R}', {}).get('zone') in ('pass', 'partial', 'inconclusive') and rh_ok:   # G(R) only where G2.3 passes or is partial
                        self.cell(f's{seed}_dev_G{R}', base + ['--method', 'g', '--R', R, '--split', 'dev', '--suite', 'hard10', '--n', NDEV, '--mult', 4])
                if imag != 'not run':
                    self.cell(f's{seed}_dev_IMAG', base + ['--method', 'g', '--R', 'inf', '--split', 'dev', '--suite', 'hard10', '--n', NDEV, '--mult', 4])
            result['seeds'].append(dict(seed=seed, cells=cells, procedure=str(proc), dev=dev_rates, token_tree_leaf=tt_choice, token_adapter=best_tok))
            write_json(RUNS / f'S5_s{seed}_summary.json', result['seeds'][-1])
        write_json(RUNS / 'Q6_RESULT.json', dict(status='completed', **result, lock=lock['lock_sha256'], elapsed_s=time.time() - self.start))
        self.stage = 'finished'; self.status('completed'); self.log(stage='Q6', event='completed')


if __name__ == '__main__':
    Q6().execute()
