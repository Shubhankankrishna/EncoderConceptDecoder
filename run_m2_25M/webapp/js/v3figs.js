/* Proposed-v3 diagrams. Unchanged v1 mechanisms reuse their code-derived figures;
   new figures describe only choices in proposed_v3.md. No v3 training result is drawn. */
const V3_SOURCE = "proposed_v3.md §§5–13 · design, untrained";

function f3Overview(A) {
  const T = A.table, P = A.params, g = el("g", {}, makeSvg(1180, 730)), svg = g.ownerSVGElement;
  frame(g, { x: 22, y: 48, w: 338, h: 590, title: `1 · Encoder · ${fmtP(T.encoder_non_embedding)}`, stroke: PAL.attn.s });
  frame(g, { x: 419, y: 48, w: 338, h: 590, title: `2 · Scratchpad · ${fmtP(T.concept_module-P.codebook)}`, stroke: PAL.conc.s });
  frame(g, { x: 816, y: 48, w: 338, h: 590, title: `3 · Decoder · ${fmtP(T.decoder)}`, stroke: PAL.ffn.s });
  box(g,{x:75,y:545,w:230,h:45,k:"emb",label:"8,192-token embedding",sub:"tied to output head",tip:`${fmtI(T.token_embedding)} stored parameters`,src:V3_SOURCE});
  box(g,{x:75,y:442,w:230,h:62,k:"attn",label:"8 layers · SSSFSSSF",sub:"d 384 · 6Q:2KV · FFN 1,056",tip:"SWA local L plus full global G. Local reach 63 tokens; full layers write only G.",src:V3_SOURCE});
  box(g,{x:75,y:343,w:230,h:53,k:"mem",label:"Engram · 8 tables",sub:"50,331,264 lookup parameters",tip:"Orders 2 and 3, 4 hash heads each, 131,071 rows × 48 dimensions.",src:V3_SOURCE});
  box(g,{x:75,y:245,w:230,h:58,k:"conc",label:"4 tokens → 1 concept C",sub:"384 dimensions · 256 per row",tip:"CRC mixes embedding, local and global streams, then concatenation-pools groups of four.",src:V3_SOURCE});
  [[75+115,545,75+115,505],[190,442,190,397],[190,343,190,304]].forEach(a=>arrow(g,[[a[0],a[1]],[a[2],a[3]]]));
  box(g,{x:474,y:533,w:230,h:44,k:"io",label:"S₀ = exact zero",tip:"The scratchpad starts empty; C is reread at every pass.",src:V3_SOURCE});
  box(g,{x:474,y:422,w:230,h:70,k:"full",label:"one shared dense block",sub:"read {C,S} → attention + FFN → S",tip:`${fmtI(P.core_layer)} parameters; gated full causal attention, NoPE; additive identity writes. No MoE or state detachment.`,src:V3_SOURCE});
  box(g,{x:474,y:310,w:230,h:55,k:"gate",label:"R2 or R4",sub:"full backprop through all passes",tip:"Ordinary batches: R2 75%, R4 25%. Batches with designated multistep tasks: R4. Inference evaluates both fixed depths.",src:V3_SOURCE});
  box(g,{x:474,y:205,w:230,h:62,k:"conc",label:"u and predicted PQ concept",sub:"48 × 128 × 8 codebook",tip:"Readout u = zcRMSNorm(S_K). The distance head predicts next-chunk codes; both u and soft predicted code vectors enter the plan.",src:V3_SOURCE});
  arrow(g,[[589,533],[589,493]]);arrow(g,[[589,422],[589,366]]);arrow(g,[[589,310],[589,268]]);
  arrow(g,[[305,275],[473,275]],{c:COL.plan,m:"ahP",w:2});
  arrow(g,[[704,238],[871,238]],{c:COL.plan,m:"ahP",w:2});
  box(g,{x:871,y:485,w:230,h:57,k:"plan",label:"target-aligned plan Pₜ",sub:"newest completed 4-token chunk",tip:"Pₜ = Wᵤuⱼ + W꜀c_pred,ⱼ + offset; j = floor((t+1)/4)−1. First 3 positions of each document get null plan.",src:V3_SOURCE});
  box(g,{x:871,y:356,w:230,h:69,k:"attn",label:"4 confined layers",sub:"only tok keys/values · window 8",tip:"Plan-aware queries cannot read another position's plan. Token-only route reaches 70 tokens.",src:V3_SOURCE});
  box(g,{x:871,y:245,w:230,h:57,k:"emb",label:"tied head Eᵀ",sub:"predict xₜ₊₁",tip:"Full vocabulary distribution; no extra output matrix.",src:V3_SOURCE});
  arrow(g,[[986,485],[986,426]]);arrow(g,[[986,356],[986,303]]);
  arrow(g,[[190,443],[190,412],[827,412],[870,412]],{c:COL.L,m:"ahL",w:1.5});
  dim(g,790,395,"tok · local path");
  box(g,{x:300,y:655,w:580,h:48,k:"loss",label:"Ltoken + λanswer Lanswer + 0.1 LNCP(final) + LVQ",sub:"proposed objective · no training-only head",tip:"Answer weight ramps from 0 to 0.1 over 300 optimizer updates; NCP is applied only at terminal depth.",src:V3_SOURCE,size:15});
  return svg;
}

function f3Scratch(A) {
  const g=el("g",{},makeSvg(1180,670)),svg=g.ownerSVGElement,P=A.params;
  txt(g,30,25,"C is read-only evidence; S is the writable scratchpad",{size:17,weight:700,anchor:"start"});
  box(g,{x:30,y:75,w:170,h:58,k:"conc",label:"C = encoder concepts",sub:"B × 256 × 384",tip:"One concept per four tokens. Its gradient remains connected to the encoder across all passes.",src:V3_SOURCE});
  box(g,{x:30,y:181,w:170,h:58,k:"io",label:"S₀ = 0",sub:"exact zero tensor",tip:"The first read is still nonzero: 0.5 × sigmoid(2) × n(C).",src:V3_SOURCE});
  frame(g,{x:252,y:75,w:562,h:468,title:`One shared LoopLayer · ${fmtI(P.core_layer)} parameters`,stroke:PAL.conc.s});
  box(g,{x:304,y:151,w:450,h:58,k:"gate",label:"Readₐ(C,S) = ½[gC·n(C) + gS·n(S)]",sub:"g = σ(U V n + b) · rank 32 · b starts +2",tip:"Separate read gates for attention and FFN. The actual read includes learned (1+γ) gains, initialized to one.",src:V3_SOURCE,size:14});
  box(g,{x:304,y:252,w:450,h:68,k:"full",label:"full causal gated attention",sub:"NoPE · document mask · per-head temperature · short conv",tip:"Read from both C and S, then additive identity write into S with a scalar gate initialized to 1.",src:V3_SOURCE,size:14});
  box(g,{x:304,y:365,w:450,h:57,k:"gate",label:"Read𝒇(C,S) → SwiGLU",sub:"384 → 1,056 → 384; add to S",tip:"The second read sees the updated scratchpad. Its write also has an identity residual path.",src:V3_SOURCE,size:14});
  arrow(g,[[200,104],[302,168]],{c:COL.plan,m:"ahP"});arrow(g,[[200,210],[252,210],[303,188]]);
  arrow(g,[[529,209],[529,251]]);arrow(g,[[529,320],[529,364]]);
  arrow(g,[[754,392],[856,392],[856,117],[815,117]],{c:COL.plan,m:"ahP",w:2});
  dim(g,860,236,"Sᵣ → next pass");
  box(g,{x:879,y:125,w:251,h:67,k:"norm",label:"terminal u = zcRMSNorm(Sₖ)",sub:"only final pass drives token loss",tip:"K is two or four. The training graph through earlier S and C remains intact.",src:V3_SOURCE,size:12.5});
  box(g,{x:879,y:244,w:251,h:66,k:"conc",label:"distance head + PQ",sub:"ĉpred = soft code expectation",tip:"48 segments × 128 codes × 8 dims; retained because the mature v1 branch helped multi-value and carry.",src:V3_SOURCE});
  box(g,{x:879,y:360,w:251,h:69,k:"plan",label:"decoder plan",sub:"Wᵤu + W꜀ĉpred + offset",tip:"The latest completed chunk is target-aligned; long-range evidence must pass through C and S.",src:V3_SOURCE});
  arrow(g,[[1005,192],[1005,243]]);arrow(g,[[1005,310],[1005,359]]);
  box(g,{x:30,y:571,w:1100,h:64,k:"white",label:"One block is reused; state changes. No expert routing, carry overwrite, detach boundary, or forced state-change loss.",sub:"Mechanical check: final token loss has nonzero gradients at all four retained state positions in a small model.",tip:"This checks the graph, not the model's future learning quality.",src:V3_SOURCE,size:14});
  return svg;
}

function f3Schedule(A) {
  const g=el("g",{},makeSvg(1180,545)),svg=g.ownerSVGElement;
  txt(g,30,29,"Proposed training depth and inference modes",{size:17,weight:700,anchor:"start"});
  frame(g,{x:26,y:67,w:545,h:385,title:"Training microbatch · ordinary",stroke:PAL.conc.s});
  box(g,{x:55,y:128,w:210,h:68,k:"conc",label:"75% → K = 2",sub:"C reread twice · full graph",tip:"One shared dense block, terminal loss only. Depth RNG checkpointed independently.",src:V3_SOURCE});
  box(g,{x:325,y:128,w:210,h:68,k:"full",label:"25% → K = 4",sub:"C reread four times",tip:"Expected ordinary depth is 2.5 passes, before forced hard-task batches.",src:V3_SOURCE});
  box(g,{x:54,y:249,w:482,h:76,k:"gate",label:"Hard-task microbatch → K = 4",sub:"lr_hop · pointer_chase · graph_walk",tip:"If any document carries a designated generator tag, the whole microbatch uses four passes. The tag affects compute only; it is not model input.",src:V3_SOURCE});
  note(g,55,369,"Mean passes = 2.5 + 1.5p, p = forced-microbatch fraction.",{size:12,color:COL.ink});
  note(g,55,394,"Report actual token-weighted depth and quality per task family.",{size:12});
  frame(g,{x:609,y:67,w:545,h:385,title:"Inference · no gold task flag",stroke:PAL.ffn.s});
  box(g,{x:647,y:128,w:210,h:68,k:"conc",label:"Normal mode · R2",sub:"fixed for every test item",tip:"Evaluate all families at R2. No gold hop count is used to select depth.",src:V3_SOURCE});
  box(g,{x:911,y:128,w:210,h:68,k:"full",label:"More compute · R4",sub:"fixed for every test item",tip:"Promote only if it earns a predeclared language or hard-task gain without regression.",src:V3_SOURCE});
  box(g,{x:648,y:249,w:474,h:76,k:"loss",label:"R8 = extrapolation diagnostic",sub:"R1 = separately trained recurrence control",tip:"Running fewer passes on R2/R4 weights is not the same control as training an R1 model.",src:V3_SOURCE});
  note(g,648,369,"Later passes earn deployment only with measured answer gains.",{size:12,color:COL.ink});
  note(g,648,394,"No per-item search using the correct test answer.",{size:12});
  box(g,{x:85,y:478,w:1010,h:44,k:"white",label:"Each pass: Sᵣ = Fθ(C,Sᵣ₋₁). Full backprop carries final answer credit through all executed passes.",size:14});
  return svg;
}

function f3Objective(A) {
  const g=el("g",{},makeSvg(1180,630)),svg=g.ownerSVGElement;
  txt(g,30,29,"Four terms, terminal depth; all targets use the full accumulated optimizer batch",{size:16,weight:700,anchor:"start"});
  const rows=[
    ["Ltoken","next-token CE","sum valid token CE / count valid tokens","Same-document successor; target ≠ PAD. EOS remains valid.","loss"],
    ["λanswer · Lanswer","correct answer spans","mean tokens → mean queries → mean synthetic documents","λ(s) = 0.1 min(1,s/300); zero when no annotated answer.","plan"],
    ["0.1 · LNCP(final)","next-concept codes","CE over 48 PQ segments and valid pairs","Current chunk nonempty; next chunk all four real; same document.","conc"],
    ["LVQ","codebook fit","squared 8-d code error; mean segments/valid chunks","Detached encoder concepts; complete real chunks only.","mem"],
  ];
  rows.forEach((r,i)=>{
    const y=83+i*115;
    box(g,{x:43,y,w:283,h:87,k:r[4],label:r[0],sub:r[1],tip:r[3],src:V3_SOURCE,size:15});
    arrow(g,[[326,y+44],[377,y+44]],{c:i===0?COL.ink:COL.plan,m:i===0?"ah":"ahP"});
    box(g,{x:378,y,w:749,h:87,k:"white",label:r[2],sub:r[3],tip:r[3],src:V3_SOURCE,size:13.2});
  });
  note(g,43,572,"NCP and VQ code vectors are detached in opposite directions; no NextLat, consistency, MoE balancing or intermediate-answer loss.",{size:12.5,color:COL.ink});
  note(g,43,595,"Answer metadata must be generated as a verified sidecar. It is not present in the current training shards.",{size:12.5,color:"#a35d18"});
  return svg;
}

function f3Evidence(A) {
  const g=el("g",{},makeSvg(1180,585)),svg=g.ownerSVGElement,m=A.measured;
  txt(g,30,29,"Evidence → decision → falsifiable check",{size:17,weight:700,anchor:"start"});
  const rows=[
    ["v1 used the plan","Recall 97.07% intact; 0.39% core-off","Keep C → S → u and the confined plan","Reproduce retrieval on a matched control"],
    ["v1 extra depth nearly flat",`TinyStories R1 ${m.v1_final_R1_tinystories.toFixed(4)}; R4 ${m.v1_final_R4_tinystories.toFixed(4)}`,"Train short K2/K4; compare trained R1","Promote R4 only for a real gain"],
    ["v2 still underperforms",`Step ${m.v2_eval_step}: val ${m.v2_val_mean.toFixed(4)}; recall ${(100*m.v2_recall_exact).toFixed(2)}%`,"Restore v1 width; remove bundled extras","B vs C vs D with equal data and LR"],
    ["v2 loop helps locally",`R1→R6 sampled TinyStories ${m.v2_sample_R1_tinystories.toFixed(4)}→${m.v2_sample_R6_tinystories.toFixed(4)}`,"Do not blame recurrence alone","Depth sweep on locked validation"],
  ];
  [27,322,617,911].forEach((x,i)=>txt(g,x+6,70,["Finding","Measured evidence","v3 response","Test"][i],{size:13,weight:700,anchor:"start"}));
  rows.forEach((r,i)=>{
    const y=93+i*100;
    r.forEach((s,j)=>box(g,{x:27+j*295,y,w:274,h:80,k:["attn","io","conc","loss"][j],label:s.length>33?s.slice(0,33)+"…":s,sub:s.length>33?s.slice(33,72):null,tip:s,src:V3_SOURCE,size:j===1?11.4:12}));
  });
  note(g,31,525,"The v2 depth numbers are 16 sampled validation rows at fixed checkpoint 1703, not the full official evaluation.",{size:12,color:COL.grey});
  note(g,31,546,"A v3 result is not shown because v3 has not been trained.",{size:12,color:COL.grey});
  return svg;
}

function f3Budget(A) {
  const g=el("g",{},makeSvg(1180,480)),svg=g.ownerSVGElement,T=A.table,M=A.proposal.macs_by_depth;
  txt(g,30,29,"Exact retained architecture budget; proposed compute is an analytical proxy",{size:16,weight:700,anchor:"start"});
  const parts=[["Encoder",T.encoder_non_embedding,"attn"],["Shared core",T.concept_module-T.codebook,"conc"],["Decoder",T.decoder,"ffn"]];
  parts.forEach((p,i)=>{
    const y=84+i*79;
    box(g,{x:40,y,w:180,h:56,k:p[2],label:p[0],sub:fmtI(p[1]),tip:"Compute parameters, excluding lookup tables and tied embedding.",src:V3_SOURCE});
    const width=650*p[1]/T.compute_params;
    el("rect",{x:243,y:y+12,width:Math.max(2,width),height:32,rx:5,fill:PAL[p[2]].s},g);
    txt(g,915,y+27,`${(100*p[1]/T.compute_params).toFixed(1)}%`,{size:15,weight:700});
  });
  box(g,{x:40,y:341,w:512,h:91,k:"mem",label:`Stored ${fmtI(T.stored_total)}`,sub:`+ Engram ${fmtI(T.engram_table)} · embedding ${fmtI(T.token_embedding)} · PQ ${fmtI(T.codebook)}`,size:13.5});
  box(g,{x:596,y:341,w:542,h:91,k:"white",label:`R2 ${(M['2'].total/1e6).toFixed(2)}M · R4 ${(M['4'].total/1e6).toFixed(2)}M MAC/token`,sub:"Measured throughput and memory are acceptance gates, not inferred from MACs.",size:13.5});
  return svg;
}

const FIGS3=[
  {id:"v3whole",t:"Whole proposal",fn:f3Overview,cap:"The retained 384-wide encoder and confined decoder surround one shared dense scratchpad block. The new answer-span term is a training weight, not an extra head."},
  {id:"v3enc",t:"Encoder layers",fn:figEncoderLayers,cap:"Retained v1 encoder: six sliding-window layers, two full layers, two streams and a 384-wide SwiGLU. Parameter figures come from the checked v1-compatible architecture."},
  {id:"v3engram",t:"Engram memory",fn:figEngram,cap:"Eight hashed n-gram tables with 50.33M lookup parameters. This mechanism is unchanged from v1."},
  {id:"v3concept",t:"Concept pooling",fn:figCRC,cap:"CRC mixes embedding, local and global streams before 4-to-1 concept pooling. The proposed core rereads these concepts every pass."},
  {id:"v3scratch",t:"Zero-state scratchpad",fn:f3Scratch,cap:"C stays available as evidence; S starts at exact zero and is updated by one shared dense layer. Every pass remains in the final token/answer gradient graph."},
  {id:"v3depth",t:"Depth & gradient",fn:f3Schedule,cap:"Two/four-pass training, fixed-depth inference and a separately trained R1 control. The hard-task flag changes training compute only."},
  {id:"v3decoder",t:"Confined decoder",fn:figDecoder,cap:"The selected confined layout is reused. The stack panel is a historical v1 control; its 265-step screen score is historical evidence, not a v3 result."},
  {id:"v3plan",t:"Plan timing",fn:figPlan,cap:"The plan uses only the latest completed chunk in the same document, preventing target-token leakage."},
  {id:"v3loss",t:"Loss & masks",fn:f3Objective,cap:"Token CE, small document-balanced answer-span CE, terminal NCP and PQ fitting. The required answer-span sidecar has not been built."},
  {id:"v3evidence",t:"Evidence → decision",fn:f3Evidence,cap:"Each proposed change is tied to measured v1/v2 behavior and a future controlled comparison. No v3 performance is implied."},
  {id:"v3budget",t:"Parameters & cost",fn:f3Budget,cap:"25.05M compute parameters, 78.58M stored parameters. The R2/R4 MAC figures are analytical proxies, not actual MPS speed."},
];
const FIG3_KEY={
  v3whole:"v3 retains v1's information route and changes the depth and training signal.",v3enc:"The 384-wide encoder preserves local token detail and global concepts.",
  v3engram:"A 50.33M-entry lookup memory remains available at low compute cost.",v3concept:"Four tokens become one 384-dimensional concept C.",
  v3scratch:"S₀ = 0. Each shared pass rereads C and writes a new S.",v3depth:"Short depth, full gradient, fixed R2/R4 inference.",
  v3decoder:"The token path reaches 70 tokens; longer evidence must use the plan.",v3plan:"Only completed chunks can contribute to the next-token plan.",
  v3loss:"Direct answer credit is small, explicit and tested separately.",v3evidence:"Historical measurements motivate decisions; only future controls can validate them.",
  v3budget:"Parameter counts are exact; speed and quality must be measured.",
};
const FIG3_3D={v3whole:"all",v3enc:"enc",v3engram:"enc",v3concept:"enc",v3scratch:"core",v3depth:"core",v3decoder:"dec",v3plan:"dec",v3loss:"all",v3evidence:"all",v3budget:"all"};
