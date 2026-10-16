"""Regenerate auditable result tables from final checkpoints and saved units."""
import datetime,json,math
from pathlib import Path
import numpy as np
from experiment_queue import JOBS
from compare_v41 import paired,main as compare_main

ROOT=Path(__file__).resolve().parent.parent
RUNS=ROOT/"runs"
def read(n,f="analysis.json"):
    p=RUNS/n/f
    return json.loads(p.read_text()) if p.exists() else None
def avg(vals):return float(np.mean(vals)) if vals else float("nan")
def fmt(x,n=4):return f"{x:.{n}f}" if x is not None and np.isfinite(x) else "—"
def agg(per):
    vals=np.array([p["mean"] for p in per]);ses=np.array([p["se"] or 0 for p in per])
    return dict(mean=float(vals.mean()),paired_item_se=float(np.linalg.norm(ses)/len(ses)),
                seed_se=float(vals.std(ddof=1)/len(vals)**.5) if len(vals)>1 else None,
                seed_min=float(vals.min()),seed_max=float(vals.max()),n_seeds=len(vals),per_seed=per)

def main():
    compare_main()
    lines=["# Saved experiment tables","", "Generated from completed artifacts at "+datetime.datetime.now(datetime.timezone.utc).isoformat()+".",
           "Partial tables remain labelled by completed seed count. Rates below are per-episode means unless otherwise stated.","",
           "## Execution inventory","", "| Run | Device / threads | Real tokens | Final | Analysis |","|---|---|---:|---|---|"]
    full_names=[j[0] for j in JOBS]+["exploratory_noplan_pin_s0_6M","exploratory_mps_plan_pin_s0_6M"]
    for name in full_names:
        j=read(name,"final.json");a=read(name)
        lines.append(f"| {name} | {j['args'].get('device','cpu')} / {j['args']['threads']} | {j['real_tokens']:,} | complete | {'complete' if a else 'pending'} |" if j else f"| {name} | — | — | pending | pending |")
    groups={"v4 / no R":[f"ref_vs_pin_v4_s{s}_6M" for s in range(3)],"v4.1":[f"v41_pin_s{s}_6M" for s in range(3)],
            "Transformer LLM":[f"ref_vs_pin_llm_s{s}_6M" for s in range(3)],"v4.1 stack":[f"v41_stack_pin_s{s}_6M" for s in range(3)],
            "v4.1 trained F loop off":[f"v41_loopoff_pin_s{s}_6M" for s in range(3)]}
    lines += ["","## P1: held-out compositions","", "| Model | Seeds | Action accuracy | Pair-second b2 | Action NTP | Long b2 | Long action NTP |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label,names in groups.items():
        js=[read(n) for n in names];js=[j for j in js if j]
        if js:lines.append(f"| {label} | {len(js)} | "+" | ".join(fmt(avg([j[s][k] for j in js])) for s,k in (("heldout","act_opt"),("heldout","act_opt_b2"),("heldout","ntp_action"),("long","act_opt_b2"),("long","ntp_action")))+" |")
    lines += ["","## P1: per-seed final numbers","", "| Model | Seed | Action accuracy | Pair-second b2 | Action NTP |","|---|---:|---:|---:|---:|"]
    for label,names in groups.items():
        for n in names:
            j=read(n)
            if j:lines.append(f"| {label} | {j['seed']} | "+" | ".join(fmt(j['heldout'][k]) for k in ('act_opt','act_opt_b2','ntp_action'))+" |")
    lines += ["","## Natural plan-lag strata","", "These are observational strata of the same held-out expert episodes. Different lags occupy different layouts/path positions, so the table does not identify a causal effect of stale plans. Averages use available per-episode values, then seeds.","",
              "| Model | Seeds | Lag | Action accuracy | b2 | Action NTP | Mean actions per episode | Episodes with b2 per seed / 200 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label,names in groups.items():
        js=[read(n) for n in names];js=[j for j in js if j]
        if not js:continue
        for lag in (1,3,5):
            # Previous reference analyses did not include the newly added lag
            # metrics; their separate audit uses identical episodes.
            vals=[];eligible=[]
            for n,j in zip([n for n in names if read(n)],js):
                section=j['heldout']
                if f'act_opt_lag{lag}' not in section:
                    audit=read(n,'phase.json')
                    if audit is None:continue
                    section={key:value['mean'] for key,value in audit.items()}
                audit=read(n,'phase.json')
                if audit:eligible.append(audit[f'act_opt_b2_lag{lag}']['n_episodes'])
                if f'act_opt_lag{lag}' in section:vals.append(section)
            if vals:lines.append(f"| {label} | {len(vals)} | {lag} | "+' | '.join(fmt(avg([v[f'{k}_lag{lag}'] for v in vals])) for k in ('act_opt','act_opt_b2','ntp_action','n'))+' | '+', '.join(str(v) for v in sorted(set(eligible)))+' |')
    lines += ["","## P2: closed-loop outcomes, 256 shared layouts per seed","",
              "| Model | Seeds | Sampled pair integrity | Greedy pair integrity | Sampled valid composition | Greedy valid composition | Sampled completion | Greedy completion |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label,names in groups.items():
        js=[read(n) for n in names];js=[j for j in js if j]
        if js:lines.append(f"| {label} | {len(js)} | "+" | ".join(fmt(avg([j[s][k] for j in js])) for s,k in (("roll_temp1","pair_ok"),("roll_greedy","pair_ok"),("roll_temp1","comp"),("roll_greedy","comp"),("roll_temp1","complete"),("roll_greedy","complete")))+" |")
    lines += ["","Pair integrity averages only layouts with an eligible transition from an even colour. Layouts without one have a saved NaN, and paired SEs use the intersection of eligible layouts. The valid-composition and completion rates above use all 256 layouts. Coverage is reported below; the zero-for-no-pair column is a secondary all-layout scoring convention, not a replacement for the registered metric.","",
              "| Model | Seed | Policy | Eligible layouts / 256 | Conditional pair integrity | All-layout score, zero for no pair |","|---|---:|---|---:|---:|---:|"]
    for label,names in groups.items():
        for n in names:
            p=RUNS/n/'per_unit.npz'
            if not p.exists():continue
            u=np.load(p);seed=read(n)['seed']
            for policy in ('temp1','greedy'):
                vals=u[f'roll_{policy}/pair_ok'];ok=np.isfinite(vals)
                lines.append(f"| {label} | {seed} | {policy} | {ok.sum()} / {len(vals)} | {fmt(float(np.nanmean(vals)))} | {fmt(float(np.nan_to_num(vals,nan=0).mean()))} |")
    lines += ["","## T1/T2: text, 5M tokens per seed","", "| Model | Seeds | Mean per-row NTP | Token-weighted NTP | recall1 exact | carry3 exact | multi4 exact |","|---|---:|---:|---:|---:|---:|---:|"]
    for label,prefix in (("v4 / no R","ref_vs_text_v4"),("v4.1","v41_text"),("Transformer LLM","ref_vs_text_llm")):
        names=[f"{prefix}_s{s}"+("_5M" if prefix=="v41_text" else "") for s in range(2)]
        js=[read(n) for n in names];js=[j for j in js if j]
        fs=[read(n,"final.json") for n in names];fs=[j for j in fs if j]
        if js:lines.append(f"| {label} | {len(js)} | {fmt(avg([j['val_ntp'] for j in js]))} | {fmt(avg([f['val_ntp'] for f in fs]))} | "+" | ".join(fmt(avg([j['probes'][k] for j in js])) for k in ('lr_recall1','lr_carry3','lr_multi4'))+" |")
    lines += ["","## Text synthetic-row pool audit","", "NTP covers entire synthetic rows, not only filler tokens. Training-pool and validation-pool items differ; their loss difference is descriptive and is not a paired memorisation test.","",
              "| Model | Seed | Training-pool row NTP | Validation-pool row NTP | Validation minus training pool |","|---|---:|---:|---:|---:|"]
    for label,prefix in (("v4 / no R","ref_vs_text_v4"),("v4.1","v41_text"),("Transformer LLM","ref_vs_text_llm")):
        for seed in range(2):
            name=f'{prefix}_s{seed}'+('_5M' if prefix=='v41_text' else '')
            j=read(name,'final.json')
            if not j:continue
            p=j['probe_row_ntp'];lines.append(f"| {label} | {seed} | {fmt(p['train_pool'])} | {fmt(p['val_pool'])} | {fmt(p['val_pool']-p['train_pool'])} |")
    lines += ["","## Fixed-checkpoint text interventions","", "Values below use the runner's token-weighted language loss. Retrieval exact-match values use 192 items per family. These interventions measure checkpoint reliance.","",
              "| Model | Seed | Mode | Language NTP | Difference from intact | recall1 | carry3 | multi4 |","|---|---:|---|---:|---:|---:|---:|---:|"]
    for label,prefix in (("v4 / no R","ref_vs_text_v4"),("v4.1","v41_text"),("Transformer LLM","ref_vs_text_llm")):
        for seed in range(2):
            name=f'{prefix}_s{seed}'+('_5M' if prefix=='v41_text' else '')
            j=read(name,'final.json')
            if not j:continue
            for mode,v in [('intact',j)]+list(j['interventions'].items()):
                lines.append(f"| {label} | {seed} | {mode} | {fmt(v['val_ntp'])} | {fmt(v['val_ntp']-j['val_ntp'])} | "+' | '.join(fmt(v['probes'][k]) for k in ('lr_recall1','lr_carry3','lr_multi4'))+' |')
    comp=read("","v41_compare.json")
    lines += ["","## Registered paired differences","", "Left minus right. The item SE follows the registered formula; seed SE is separate.","",
              "| Comparison | Metric | Difference | Paired item SE | Training-seed SE | Seed range | Seeds |","|---|---|---:|---:|---:|---|---:|"]
    for name,result in comp.items():
        if name.startswith("fresh_") or "fresh_LLM" in name:continue
        for k in ("heldout/act_opt","heldout/act_opt_b2","heldout/ntp_action","roll_temp1/pair_ok","roll_greedy/pair_ok","row_ntp"):
            v=result["metrics"].get(k)
            if v:lines.append(f"| {name} | {k} | {fmt(v['mean'])} | {fmt(v['paired_item_se'])} | {fmt(v['seed_se'])} | {fmt(v['seed_min'])} … {fmt(v['seed_max'])} | {v['n_seeds']} |")
    lines += ["","## Fixed-checkpoint interventions on pinpad","", "These measure checkpoint reliance, not training-time necessity.","",
              "| Model | Seeds | Intervention | Action accuracy | b2 | Action NTP |","|---|---:|---|---:|---:|---:|"]
    for label in ("v4.1","v4.1 stack"):
        js=[read(n) for n in groups[label]];js=[j for j in js if j]
        if not js:continue
        for iv in ("intact","core_off","plan_swap","M1","loop_off","u_off","c_off","core_shared_off","enc_shared_off","R_local","core_off_R_local"):
            vals=[j["heldout"] if iv=="intact" else j["interventions"].get(iv) for j in js]
            vals=[v for v in vals if v]
            if vals:lines.append(f"| {label} | {len(vals)} | {iv} | "+" | ".join(fmt(avg([v[k] for v in vals])) for k in ('act_opt','act_opt_b2','ntp_action'))+" |")
    secondary={}
    for prefix,comparison in (('ref_vs_pin_llm','v41_minus_LLM'),('ref_vs_pin_v4','v41_minus_noR')):
        for policy in ('temp1','greedy'):
            per=[]
            for seed in range(3):
                pa=RUNS/f'v41_pin_s{seed}_6M/per_unit.npz';pb=RUNS/f'{prefix}_s{seed}_6M/per_unit.npz'
                if not pa.exists() or not pb.exists():continue
                a,b=np.load(pa),np.load(pb);key=f'roll_{policy}/pair_ok'
                per.append(dict(seed=seed,**paired(np.nan_to_num(a[key],nan=0),np.nan_to_num(b[key],nan=0))))
            if per:secondary[f'rollout_all_layout/{comparison}/{policy}']=agg(per)
    lines += ["","## Calibrated plans: paired effects relative to intact","", "Constants are estimated from an independent training bank; nulls and offsets stay legal.","",
              "| Model / data | Mode | Metric | Difference | Paired item SE | Seed SE | Seeds |","|---|---|---|---:|---:|---:|---:|"]
    for data,label,names,metrics in (("pinpad","v4.1 / pinpad",groups['v4.1'],('act_opt','act_opt_b2','ntp_action')),
                              ("noR_pinpad","v4 / pinpad",groups['v4 / no R'],('act_opt','act_opt_b2','ntp_action')),
                              ("stack_pinpad","stack / pinpad",groups['v4.1 stack'],('act_opt','act_opt_b2','ntp_action')),
                              ("loopoff_pinpad","loop off / pinpad",groups['v4.1 trained F loop off'],('act_opt','act_opt_b2','ntp_action')),
                              ("text","v4.1 / text",[f"v41_text_s{s}_5M" for s in range(2)],('row_ntp',))):
        for mode in ("constant","zero_content","null"):
            for metric in metrics:
                per=[]
                for n in names:
                    p=RUNS/n/"plan_diagnostic_units.npz"
                    if not p.exists():continue
                    u=np.load(p);z=paired(u[mode+'/'+metric],u['intact/'+metric]);per.append(dict(seed=read(n,'final.json')['args']['seed'],**z))
                if per:
                    v=agg(per);secondary[f'{data}/{mode}/{metric}']=v
                    lines.append(f"| {label} | {mode} | {metric} | {fmt(v['mean'])} | {fmt(v['paired_item_se'])} | {fmt(v['seed_se'])} | {len(per)} |")
    for mode in ('constant','zero_content','null'):
        for metric in ('act_opt','act_opt_b2','ntp_action'):
            per=[]
            for seed in range(3):
                pa=RUNS/f'v41_pin_s{seed}_6M/plan_diagnostic_units.npz'
                pb=RUNS/f'ref_vs_pin_v4_s{seed}_6M/plan_diagnostic_units.npz'
                if pa.exists() and pb.exists():
                    a,b=np.load(pa),np.load(pb)
                    za=a[mode+'/'+metric]-a['intact/'+metric]
                    zb=b[mode+'/'+metric]-b['intact/'+metric]
                    per.append(dict(seed=seed,**paired(za,zb)))
            if per:secondary[f'calibrated_interaction/R_minus_noR/{mode}/{metric}']=agg(per)
    lines += ["","## Counterfactual facts, fixed 128 legal prefix pairs","",
              "| Model | Seed | Original accuracy | Changed-fact accuracy | Prediction changes | Changed correct given original correct |","|---|---:|---:|---:|---:|---:|"]
    for label,names in list(groups.items())+[("MPS no plan",['exploratory_noplan_pin_s0_6M']),("MPS normal plan",['exploratory_mps_plan_pin_s0_6M'])]:
        for n in names:
            j=read(n,'counterfactual.json')
            if j:
                v=j['intact'];seed=read(n,'final.json')['args']['seed']
                lines.append(f"| {label} | {seed} | "+" | ".join(fmt(v[k]) for k in ('original_accuracy','changed_accuracy','prediction_changes','changed_accuracy_given_original_correct'))+" |")
    for metric in ('original_correct','changed_correct'):
        per=[]
        for s in range(3):
            pa=RUNS/f'v41_pin_s{s}_6M/counterfactual_units.npz';pb=RUNS/f'ref_vs_pin_llm_s{s}_6M/counterfactual_units.npz'
            if pa.exists() and pb.exists():
                a,b=np.load(pa),np.load(pb);z=paired(a['intact/'+metric].astype(float),b['intact/'+metric].astype(float));per.append(dict(seed=s,**z))
        if per:secondary['counterfactual/v41_minus_LLM/'+metric]=agg(per)
    normal,noplan='exploratory_mps_plan_pin_s0_6M','exploratory_noplan_pin_s0_6M'
    if read(normal) and read(noplan):
        a,b=np.load(RUNS/noplan/'per_unit.npz'),np.load(RUNS/normal/'per_unit.npz')
        for k in ('heldout/act_opt','heldout/act_opt_b2','heldout/ntp_action','roll_temp1/pair_ok','roll_greedy/pair_ok'):
            secondary['matched_MPS/noplan_minus_normal/'+k]=paired(a[k],b[k])
    lines += ["","## Coarse codes and causally legal probes","",
              "The NMI label is the majority subgoal within a chunk; it is non-causal. The permutation null preserves each episode's temporal structure.","",
              "| Model | Seed | Codes used | NMI | Null mean | Null tail p | Core pass 1 subgoal probe | Core pass 2 subgoal probe | Majority |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label,names in [("v4.1",groups['v4.1']),("v4.1 stack",groups['v4.1 stack']),("MPS no plan",[noplan]),("MPS normal plan",[normal])]:
        for n in names:
            j=read(n)
            if not j:continue
            c=j.get('code_nmi',{});p=j.get('probes',{})
            lines.append(f"| {label} | {j['seed']} | {c.get('codes_used','—')} | {fmt(c.get('nmi'))} | {fmt(c.get('episode_relabel_null_mean'))} | {fmt(c.get('episode_relabel_tail_p'))} | {fmt(p.get('core_pass1_completed',{}).get('sub_heldout'))} | {fmt(p.get('core_pass2_completed',{}).get('sub_heldout'))} | {fmt(p.get('_majority',{}).get('sub_heldout'))} |")
    ctrl=read('v41_pin_s0_6M','control.json')
    if ctrl:
        lines += ["","## Frozen oracle control, all learning rates","", "128 prefixes: 52 initially wrong, 76 initially right. Ten steps, weights frozen.","",
                  "| Site / learning rate | Wrong fixed | Right broken | Optimal-set probability gain | Random wrong fixed | Oracle step norm | Random achieved step norm |","|---|---:|---:|---:|---:|---:|---:|"]
        for key,v in ctrl['sites'].items():
            o,r=v['oracle'],v['random']
            lines.append(f"| {key} | {fmt(o['fraction_wrong_fixed'])} | {fmt(o['fraction_right_broken'])} | {fmt(o['optimal_probability_gain'])} | {fmt(r['fraction_wrong_fixed'])} | {fmt(v['oracle_step_norm'])} | {fmt(v['random_achieved_step_norm'])} |")
    (RUNS/'secondary_compare.json').write_text(json.dumps(secondary,indent=2,allow_nan=False)+'\n')
    (ROOT/'RESULT_TABLES.md').write_text('\n'.join(lines)+'\n')
    print(ROOT/'RESULT_TABLES.md')

if __name__=='__main__':main()
