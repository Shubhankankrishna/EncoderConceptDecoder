/* Proposed v3 overview and figures. Static design content, with historical measurements labeled. */
(function () {
  let A3=null, figsDone=false;
  const m=n=>(n/1e6).toFixed(2)+"M";
  const stat=(label,value,sub)=>`<div class="card stat"><div class="l">${label}</div><div class="v">${value}</div><div class="s">${sub}</div></div>`;

  function renderOverview() {
    const A=A3,T=A.table,P=A.proposal,H=A.measured;
    document.getElementById("v3-stats").innerHTML=[
      stat("compute parameters",m(T.compute_params),"exact v1-compatible model count"),
      stat("stored parameters",m(T.stored_total),"including Engram and tied embedding"),
      stat("token / concept width",A.ecfg.d_model,"6 query heads · 2 KV heads"),
      stat("scratchpad start","S₀ = 0","C reread each pass"),
      stat("ordinary depth","R2 / R4","75% / 25% microbatches"),
      stat("decoder reach",A.reach_dec+" tokens","long-range path uses the plan")
    ].join("");
    document.getElementById("v3-hero-side").innerHTML=`<div class="card" style="height:100%">
      <div class="l" style="font-size:12px;color:var(--ink3);text-transform:uppercase;letter-spacing:.08em;font-weight:650">Proposal status</div>
      <div style="font-size:22px;font-weight:700;margin:6px 0 8px">Design reviewed · untrained</div>
      <p style="margin:0 0 12px;font-size:13px">The 3D pass animation illustrates the proposed graph. It does not display learned v3 activations or predicted performance.</p>
      <div class="proposal-status"><b>Primary test:</b> compare the new recipe with a corrected v1 control on identical tokenized data and optimizer steps. A v1-level recovery alone does not establish a new capability.</div>
      <div class="v3-kv" style="margin-top:14px"><span>frozen evidence</span><b>29 Sep 2026</b><span>review</span><b>83 checks passed</b><span>training</span><b>not started</b></div></div>`;
    document.getElementById("v3-evidence").innerHTML=`<table class="t"><thead><tr><th>Observed result</th><th>What it establishes</th><th>v3 design response</th></tr></thead><tbody>
      <tr><td>v1 final test recall <b>${(100*H.v1_recall_exact).toFixed(2)}%</b>; core-off <b>0.39%</b></td><td>The concept route carries long-range facts.</td><td>Keep the v1-width encoder, continuous u route, PQ branch and confined decoder.</td></tr>
      <tr><td>v1 final TinyStories R1 <b>${H.v1_final_R1_tinystories.toFixed(4)}</b>, R4 <b>${H.v1_final_R4_tinystories.toFixed(4)}</b> nats</td><td>The trained v1 got little benefit from extra passes.</td><td>Use short depth and require separately trained R1 plus fixed R2/R4 comparisons.</td></tr>
      <tr><td>Near equal training time: v1 step ${H.v1_equal_time_step} validation <b>${H.v1_equal_time_val_mean.toFixed(4)}</b>; v2 step ${H.v2_eval_step} <b>${H.v2_val_mean.toFixed(4)}</b></td><td>v2 underperforms as a system. Width, depth, losses and schedule are confounded.</td><td>Restore proven capacity and isolate hygiene, depth and answer weighting in B/C/D arms.</td></tr>
      <tr><td>At v2 step ${H.v2_eval_step}, sampled TinyStories R1→R6 improves <b>${H.v2_sample_R1_tinystories.toFixed(4)}→${H.v2_sample_R6_tinystories.toFixed(4)}</b></td><td>Loops can improve token loss at fixed weights; more passes have diminishing returns.</td><td>Keep a dense scratchpad, measure whether R4 earns its cost.</td></tr>
    </tbody></table><p class="muted" style="font-size:12px;margin:9px 0 0">v1 final-test and near-equal-time validation are different evaluations. The v2 R sweep uses 16 sampled rows, not the full official validation set. None are v3 results.</p>`;
    const story=(cls,n,title,tag,text,facts)=>`<div class="story-card ${cls}"><div class="story-num">${n}</div><h4>${title} <span>${tag}</span></h4><p>${text}</p><div class="facts">${facts.map(x=>`<span>${x}</span>`).join("")}</div></div>`;
    document.getElementById("v3-story").innerHTML=
      story("c-enc",1,"Encoder","v1 capacity retained",`Eight ${A.ecfg.layer_pattern} layers keep local token detail and a global concept stream. Engram reads 8 hashed tables, then CRC mixes streams and pools 4 tokens into each 384-d concept.`,[m(T.encoder_non_embedding)+" compute",m(T.engram_table)+" Engram","reach 63","256 concepts / row"])+
      '<div class="story-arrow">→</div>'+story("c-core",2,"Scratchpad","zero S, shared weights",`One dense causal block repeatedly reads C and S, then adds gated attention and SwiGLU updates to S. Terminal u and predicted PQ codes make the decoder plan. Full gradients traverse two or four passes.`,[m(T.concept_module-T.codebook)+" core","S₀ = 0","R2/R4","48 × 128 × 8 PQ"])+
      '<div class="story-arrow">→</div>'+story("c-dec",3,"Decoder","confined, plan-aware",`Four decoder layers use position-private plan-aware queries, with keys and values from local tok only. The tied output head predicts the next token; long-range information must pass through the plan.`,[m(T.decoder)+" compute","4 layers","reach 70","8,192-token head"]);
    document.getElementById("v3-depth").innerHTML=`<h4>How S is updated</h4><p><code>S₀ = 0</code>. At each pass the same dense block reads <code>{C,S}</code>, applies document-causal full attention and a 384→1,056→384 SwiGLU, and adds both gated updates to S. C is reread with a live gradient path.</p><p>Ordinary microbatches draw <b>R2 75%</b> and <b>R4 25%</b>. A batch containing a designated multistep family uses R4. The task tag affects training compute; both fixed depths are tested on every family at inference.</p><p class="muted">R8 is an extrapolation diagnostic. A separately trained R1 model tests whether recurrence earns its cost.</p>`;
    document.getElementById("v3-objective").innerHTML=`<h4>Exact proposed objective</h4><p><code>L = Ltoken + λanswer(s)Lanswer + 0.1 LNCP(final) + LVQ</code></p><ul>
      <li>Token CE: all valid next tokens, counted across the full accumulated optimizer batch.</li>
      <li>Answer CE: mean tokens within span → mean queries within document → mean annotated synthetic documents. <code>λ=0.1 min(1,s/300)</code>.</li>
      <li>NCP: final-depth next-concept codes on real, same-document pairs only.</li>
      <li>VQ: complete real chunks train the codebook through detached encoder concepts.</li></ul><p class="muted">No new answer head, NextLat, MoE balance or state-consistency loss.</p>`;
    document.getElementById("v3-data").innerHTML=`<h4>Data and training budget</h4><ul>
      <li>Keep the existing 8,192-token vocabulary and 45% language / 22% symbolic / 13% retrieval / 10% code / 10% long-range-probe mixture.</li>
      <li>Recreate answer-span sidecars from task generators and require byte-identical token/doc arrays. If regeneration differs, version the dataset and rerun controls.</li>
      <li>AdamW (0.9, 0.95), 300 warm-up steps to 6e-4, one cosine to 6e-5, global clip 1.</li>
      <li>B2 × accumulation 16 × 1,024; 4,050 steps = <b>${fmtI(P.slot_budget)}</b> processed slots.</li></ul>`;
    document.getElementById("v3-gates").innerHTML=`<h4>Accept or reject v3</h4><ol>
      <li>Neither TinyStories nor WikiText may worsen more than 0.03 nats versus corrected control B at equal data budget.</li>
      <li>Target ≥5 percentage points on the declared hard-task aggregate, with recall losing ≤2 points; publish each family and locked-test confirmation.</li>
      <li>Steady training throughput ≥90% of B, unless equal-time quality is better; measure memory and checkpoint costs.</li>
      <li>Promote R4 only for ≥0.01 nats language or ≥5-point hard-task gain without material regression. Otherwise use R2 or trained R1.</li></ol><p class="muted">These are engineering targets, not forecasts.</p>`;
    document.getElementById("v3-experiments").innerHTML=`<table class="t"><thead><tr><th>Arm</th><th>Architecture</th><th>Hygiene</th><th>Depth</th><th>Answer term</th><th>Purpose</th></tr></thead><tbody>
      ${P.comparison_arms.map(([a,name,mask,depth,answer])=>`<tr><td><b>${a}</b></td><td>${name}</td><td>${mask}</td><td>${depth}</td><td>${answer}</td><td>${({A:"historical reference",B:"fair corrected control",C:"depth effect",D:"answer effect"})[a]}</td></tr>`).join("")}
      </tbody></table><p class="muted" style="font-size:12px;margin:9px 0 0">B and D are the first resource-limited comparison. C attributes a detected difference. A trained fixed-R1 control tests recurrence separately. Warm-start results count their inherited v1 training.</p>`;
  }

  function goto(id) {
    if(!document.getElementById("tab-v3arch2d").classList.contains("on"))showTab("v3arch2d");
    document.querySelectorAll("#fig3-nav button").forEach(b=>b.classList.toggle("on",b.dataset.id===id));
    const f=document.getElementById("fig3-"+id); if(f)f.scrollIntoView({block:"start"});
  }
  function open3D(view) {
    showTab("v3arch3d"); window.scrollTo(0,0);
    let n=0;const timer=setInterval(()=>{const b=document.querySelector(`#v3c-bar [data-view="${view}"]`);
      if((b&&document.querySelector("#v3c-info h5"))||++n>40){clearInterval(timer);if(b)b.click();}},100);
  }
  function renderFigs() {
    if(figsDone)return;figsDone=true;
    const nav=document.getElementById("fig3-nav"),box_=document.getElementById("figs3");
    const ctx={figs:FIGS3,go:goto,open3d:open3D,key:FIG3_KEY,map3d:FIG3_3D};
    FIGS3.forEach((f,i)=>{
      const b=document.createElement("button");b.dataset.id=f.id;b.innerHTML=`<span class="fn">${i+1}</span>${f.t}`;if(!i)b.classList.add("on");b.onclick=()=>goto(f.id);nav.appendChild(b);
      const fig=document.createElement("figure");fig.className="fig";fig.id="fig3-"+f.id;
      const h=document.createElement("h3");h.className="figt";h.innerHTML=`<span class="figt-n">${i+1}</span>${f.t}`;fig.appendChild(h);
      try{fig.appendChild(f.fn(A3));}catch(e){const p=document.createElement("p");p.className="muted";p.textContent="figure error: "+e.message;fig.appendChild(p);console.error(e);}
      const cap=document.createElement("figcaption");cap.textContent=f.cap;fig.appendChild(cap);box_.appendChild(fig);
      figTools(fig,f.id,i,FIGS3.length,ctx);
    });
    window.addEventListener("scroll",()=>{if(!document.getElementById("tab-v3arch2d").classList.contains("on"))return;
      let cur=null;document.querySelectorAll("#figs3 figure.fig").forEach(f=>{if(f.getBoundingClientRect().top<innerHeight*.4)cur=f.id.slice(5);});
      if(cur)document.querySelectorAll("#fig3-nav button").forEach(b=>b.classList.toggle("on",b.dataset.id===cur));},{passive:true});
  }
  async function ensure(){if(!A3){const r=await fetch("data/arch_v3.json");if(!r.ok)throw new Error("v3 architecture data unavailable");A3=await r.json();window.ARCH3=A3;}return A3;}
  async function shown(id){if(!id.startsWith("v3"))return;try{await ensure();if(id==="v3overview"&&!document.getElementById("v3-stats").innerHTML)renderOverview();if(id==="v3arch2d")renderFigs();}catch(e){console.error("v3 page",e);}}
  window.addEventListener("tabshown",e=>shown(e.detail));
  document.addEventListener("click",e=>{const a=e.target.closest("[data-fig3]");if(a){e.preventDefault();ensure().then(()=>{renderFigs();goto(a.dataset.fig3);});}});
  const initial=(document.querySelector("section.tab.on")||{}).id;if(initial&&initial.startsWith("tab-v3"))shown(initial.slice(4));
})();
