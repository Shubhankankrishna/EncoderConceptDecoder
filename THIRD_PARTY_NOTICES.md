# Third-party notices and literature

The root MIT license covers original EncoderConceptDecoder code. It does not replace third-party licenses or grant rights to datasets and papers.

## Three.js

Preserved web dashboards vendor Three.js and its example helpers. The module header identifies revision 169 and states `SPDX-License-Identifier: MIT`, copyright 2010–2024 Three.js Authors. The [upstream MIT license](third_party/threejs/LICENSE) is retained alongside the repository notices. Source headers remain unchanged. Upstream: [Three.js](https://github.com/mrdoob/three.js/tree/r169).

## Papers, datasets, and upstream research material

### NCM and BERT assets used by Astra

Astra M2/V3/V4 depend on [HazyResearch/ncm](https://github.com/HazyResearch/ncm), pinned to `c70274273f489abc73832b2b98584a8483b6b203`. The temporary checkout stays local; its original [MIT license](third_party/ncm/LICENSE), copyright 2025 Hermann Kumbong, is retained. Frozen upstream source copies already captured inside run `source/` evidence directories are preserved with that attribution and license; they are not claimed as original project code. Upstream configuration files and source provenance retain their attribution. The [codec manifest](v15_astra_m2/artifacts/codec/manifest.json) pins the separate official weights repository and revision; this project does not redistribute those weight tensors.

Small BERT tokenizer files are preserved with [their provenance](v15_astra_m2/artifacts/codec/lm1b/bert-tokenizer/provenance.json). They originate from [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) and use the [Apache 2.0 license](third_party/bert-tokenizer/LICENSE). The license copy comes from the [original Google BERT source](https://github.com/google-research/bert/blob/master/LICENSE). These assets are not relicensed by the root MIT file.

Dataset bodies, graph fixtures, FineWeb document text and prepared evaluation examples remain local. Public manifests, generators, aggregate results and provenance records do not grant dataset redistribution rights.

Downloaded PDF copies and temporary upstream source checkouts are excluded from publication and indexed locally by path/checksum. Research notes and architecture documents preserve their citations. Consult the [historical paper catalog](repo_review_2026-10-13/PAPERS.md), [architecture paper catalog](JEPA_Architecture_Papers_Catalog.md), [V10 dataset provenance](v10_10M/DATASETS.md), and [V13 dataset protocol](Dataset_v13.md) for sources and restrictions.

Historical documents may link to a local paper filename that is absent from a clone. Use the public citation in the document or paper catalog. Do not interpret those local links as public download links. Copied project-source audit snapshots retain their original provenance and are not claimed as independent work.
