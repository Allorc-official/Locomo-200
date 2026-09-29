# NOTICE — attribution, modifications, and licensing

## Dataset

`dataset/locomo200.json` and `dataset/locomo200.manifest.json` are derived from
**LoCoMo** — "Evaluating Very Long-Term Conversational Memory of LLM Agents" by Adyasha
Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang
(ACL 2024).

- Upstream repository: https://github.com/snap-research/locomo
- Upstream file: `data/locomo10.json` (SHA-256
  `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`)
- Upstream license: **Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0)**, https://creativecommons.org/licenses/by-nc/4.0/
- This package is an **independent, filtered derivative**: it is not affiliated with or
  endorsed by the LoCoMo authors, and the original dataset, code, and updates remain at
  the upstream repository.

### Modifications (required notice)

This package redistributes an **adapted subset** of the upstream dataset:

- Only the 10 conversations from the upstream release are used, and only 200 of their
  questions (out of 1,986) are retained.
- Category 3 (open-domain / world-knowledge) questions are excluded (96).
- Category 1 questions without cross-session evidence are excluded (12); category 5
  questions without null answers are excluded (2); questions with unresolved evidence
  references are excluded (6).
- The remaining eligible questions are sampled evenly by evidence difficulty to 20 per
  conversation: 10 multi-hop, 7 temporal, 2 adversarial, 1 single-hop.
- Conversations are relayed to the agent as ordered, text-only session transcripts
  (speaker + text); image URLs and captions are not sent to the agent.
- Evidence turns, difficulty metadata, and the selection manifest are added for scoring.

No changes were made to the conversation content. The filtered dataset file has SHA-256
`411ec27105f3c7b632b9b509f9e25149c7c937cf971c053fb3c7e863a6627a5c`.

Because the upstream material is non-commercial, this dataset (and any redistribution of
it, including this package's `dataset/` directory) may only be used for
**non-commercial purposes**. The MIT license below applies to the **code only** and does
not apply to the dataset files.

## Code

`runner/` and `judge/` in this package are original code. They are distributed under the
MIT license in `LICENSE`. If you republish this package, keep this notice with the dataset
files and preserve the attribution and license references above.
