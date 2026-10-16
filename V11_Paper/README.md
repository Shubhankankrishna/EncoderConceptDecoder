# V11 design source packet

Use with the root **`V11_Architecture_Design_Prompt.md`**. Give the same prompt and evidence independently to the three intended models so their architectural choices can be compared under the same constraints. The brief requests a design and review, not a training launch.

## Supplied papers

| File | Main passages for this task |
|---|---|
| `Heirachical latent prediction Microsoft.pdf` | HiLP §2, pp. 1–2; Algorithm 1 p. 7; costs p. 8 |
| `Next-Latent Prediction Transformer.pdf` | NextLat §3, pp. 4–6; Algorithm 1 p. 25; objective ablations pp. 26–28 |
| `NCP-ArchPreview Technical Report.pdf` | ECD, quantization, CRC/IRC, shift, and losses, §§2–3, pp. 4–9 |
| `Explorative Modeling.pdf` | Objectives and matching, §3, pp. 6–9; implementation in Appendix C |
| `Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning.pdf` | Controller discovery and internal RL, pp. 5–7; Appendix D |

Page numbers refer to PDF pages. Files are copies with their supplied filenames unchanged.

## V10.3 evidence

`V10_3_Evidence/` contains the root V10.3 design and tiny experiment report; the tiny specification, detailed tables, comparison JSON, test results, layout audit, and sleep results; and selected source files under `source/` preserving project-relative paths.

Start with the prompt's evidence summary and `V10_3_tiny_REPORT.md`, then inspect `v10_3_tiny/comparison.json` and `v10_3_tiny/runs/sleep_X8_s1/sleep.json`. Consult `source/` to resolve implementation details, especially the pooler, two streams, code-space offset, candidate objective, correction bank, and gate smoother. That subset is not a runnable training distribution: datasets, checkpoints, dependencies, and some imported utilities are intentionally absent.

The root V10.3 design diagnoses older V10.1 checkpoints. The `v10_3_tiny` artifacts describe the actual tiny successor experiments. Do not merge their parameter counts, tokenizers, or measurements. Historical reports include interpretations that the prompt qualifies; the new review does not rewrite those original records. Some links inside copied historical documents point to artifacts outside this compact packet.

Additional primary online references are linked in the prompt: V-JEPA 2, Anthropic's global workspace work, Kimi Attention Residuals, DeepSeek mHC, and Qwen. These are not among the five supplied PDF files. Unverified version-specific claims must remain marked as uncertain.

`manifest.json` records the original source, packet-relative copy, byte size, and SHA-256 for each of 30 copied files. Source content was preserved. No model output or V11 experiment result is included.
