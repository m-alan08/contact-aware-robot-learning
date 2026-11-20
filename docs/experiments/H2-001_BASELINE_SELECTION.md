# H2-001 — Baseline selection audit

*Phase 1 of H2-001. Audit date: 2026-11-18. Author: Claude (Opus 5.5) for the
project team. Nothing in this document is an experimental result.*

**Outcome: the execution gate is not passed.** No public learned tactile policy
meets all four requirements:

* a predicted-contact pathway;
* a live-tactile pathway;
* released checkpoints;
* an evaluation environment we can run.

So, as the H2 brief requires, we stopped before reproduction. No protocol was
registered, no experiment was run, and no replacement architecture was built.
§6 lists the options and what each needs.

## 1. What H2 needs from a baseline

H2 (wording unchanged from the plan, §3):

> Under recoverable contact disturbances, selective intervention on an outdated
> forecast improves recovery when live tactile measurements are informative.

The plan (§2.2, §4.2) and the H2-001 brief require the following. The criteria
column uses these codes in the §3 table.

| Requirement | Why | Criterion |
|---|---|---|
| A *learned* policy that uses predicted contact at execution time | H1-002 showed that, in our task, a 2-parameter analytic forecaster explains all the anticipation benefit. H2 must study a learned controller that actually depends on its forecasts. | R1 |
| The same policy uses live tactile feedback | H2 concerns disagreement between prediction and measurement | R2 |
| Forecast and feedback pathways can be intervened on separately | The central experiment holds live touch fixed and changes only the forecast | R3 |
| Pretrained checkpoint available | Reproduce before intervening; avoid building a large model ourselves | R4 |
| Training and evaluation data available | Fine-tuning, forecast-error measurement, and OOD checks | R5 |
| An environment we can evaluate in: a simulator, or our own hardware | Paired, repeatable trials with controlled disturbances | R6 |
| Compute we can obtain | See §4 | R7 |

## 2. Sources checked

* The literature review (`docs/literature/literature_review.pdf`), refs. [3]–[14].
* For each paper:
  * the arXiv abstract and HTML pages;
  * the project page;
  * GitHub: the repository and README, recent commits, and releases;
  * Hugging Face Hub searches for models and datasets.
* We also searched GitHub for each name, because some repositories are not
  linked from their paper.
* Everything was checked on 2026-11-18. Availability can change, so re-check
  before acting on this document.

## 3. Candidates

Key: ✓ = verified available; ✗ = verified absent, or not found after the
searches in §2; ~ = partial. Code and checkpoints count as present only if a
public link exists.

| Candidate | Code | Checkpoint | Data | Sim | Forecast | Live touch (R2) | Separable (R3) | Notes |
|---|---|---|---|---|---|---|---|---|
| **TacPAC** [12] (`LogosIntelligenceLab/TacPAC`, MIT, head `1474162`, 2026-10-24) | ✓ model, training, deployment, CPU unit tests | ✗ README: "Datasets and checkpoints are being prepared for public release" | ✗ private Flexiv datasets | ✗ real robot only (Flexiv, 5 tasks) | ✓ **latent** (Wan2.2 VAE tactile-image latents, produced by the video expert) | ✓ the tactile expert reads each new tactile frame | **~** see §3.1 | Best conceptual fit. Uses a Wan2.2-TI2V-5B backbone. Training defaults to 8 GPUs, and it needs flash-attn (CUDA). |
| **T-Rex** [14] (`ZhuoyangLiu2005/T-Rex`, MIT) | ✓ post-training and inference | ✓ pretrain and midtrain on HF (`miniFranka/…`) | ✓ about 50 h in LeRobot v3 (`zekaiwang/trex_dataset`) | ✗ real bimanual Dexmate Vega-1 with Sharpa hands | ✗ **none**. Reactive refinement from current touch plus a cached VL context. Not a future forecast (also stated in plan §1.3). | ✓ | n/a | A strong *reactive* comparison for later. It cannot be the H2 subject, because there is no forecast to intervene on. Qwen3-VL-2B on CUDA. Running it needs the same hardware. |
| **VT-WAM** [7] | ✗ no repo found. The project page links no code. | ✗ | ✓ real-robot zarr for 6 tasks (`ShuaiTian/VT-WAM-OXT`) | ✗ | ✓ deformation prediction | ✓ | unknown | Data only |
| **Tactile-WAM** [9] (`wusiyu66/Tactile-WAM`) | ~ architecture only (32 KB repo) | ✗ README: "Pretrained weights and data are not included" | ✗ | ~ the paper evaluates on ManiFeel, but no environment code is released | ✓ touch-aware bias from predicted touch | ✓ | unknown | Not runnable |
| **Dream-Tac** (`LYFCLOUDFAN/Dream-Tac`, Apache-2.0; a TacPAC inspiration) | ✓ (Cosmos Policy based) | ✗ | ✗ | ✗ real Franka | ✓ | ✓ contact gate | unknown | Cosmos-Predict2-2B, 8-GPU training |
| **ViTaL** [5] (`yilin-wu98/vital`) | ✓ | ~ README: weights "supplied separately". No public link found. | ✓ 980 episodes, 716 GB | ✗ real robot | ✓ tactile world model | Used for *steering* candidate actions, not as an in-loop policy input | n/a | It is a policy-steering method, not a policy whose forecast feeds the action |
| **ForeTac** [6], **ForeTac-VLA** [3], **TouchWorld** [11], **PredTac**, **TacDyn-WAM**, **OmniVTA**, **Contact-Grounded Policy**, **Aux. tactile dynamics DP** [13] | ✗ no runnable code found. OmniVTA's repo contains only a LICENSE. The others have project pages only, or nothing. | ✗ | ~ OmniVTA links a dataset | PredTac, TacDyn-WAM, and Contact-Grounded Policy report simulation, but no environment code is released | ✓ (mostly) | ✓ | unknown | Not runnable |
| **Reactive Diffusion Policy** [10] | ✓ | ✓ HF (`WendiChen/…`) | ✓ | ✗ real robot | ✗ slow/fast reactive, no tactile forecast | ✓ | n/a | Reactive comparison only |
| **ManiFeel** benchmark (`purdue-mars/manifeel`, MIT) and third-party checkpoints (e.g. `guitarorangedog/manifeel-dp-gear`, `…-tavla-gear`, `…-tavla-bulb`; LeRobot 0.6.1) | ✓ | ✓ diffusion-policy and TA-VLA checkpoints from a third party (card example: DP on `gear` reports 46.0% success, 4 seeds × 50 episodes) | ✓ (`purdue-mars/manifeel`) | ✓ **Isaac Gym / TacSL** (simulated GelSight) | ✗ the baseline policies do not forecast | ✓ | n/a | The only *runnable, simulated, learned tactile policy* with checkpoints. There is no forecast pathway to intervene on. Needs Linux and an NVIDIA GPU. |

### 3.1 TacPAC's forecast and feedback pathways (from reading the code; nothing was run)

TacPAC was cloned into a scratch directory at `1474162`. Only these two files
were read:
* `TacPAC/model/framework/WM4A/WanMoTJointTacExpert.py`
* `TacPAC/model/modules/wan_mot/tactile_expert.py`

How the cache is built and used:

* **The cache is concatenated from two parts** (`_prefill_tac_cache`, lines
  102–169). One clean joint pass over the denoised video and the plan stores,
  for each layer:
  * the K/V of the **predicted-tactile token slice** (`layer_video[k][:, tac_idx]`);
  * the K/V of the **action tokens**.
* **The live tactile frame enters only as queries.** In `TactileExpert.forward`,
  the queries are `[fresh tactile tokens, action tokens]`. The keys and values
  are the cache plus the expert's own tokens. The output is delta actions for
  the unexecuted suffix.

What this means for H2:

* **R3 is partly met.** The predicted-tactile K/V can be replaced, masked, or
  refreshed independently of the live-tactile queries. This is the
  "hold feedback, change forecast" intervention H2 needs.
* **Entanglement.** The *action* K/V in the cache come from Mixture-of-Transformers
  joint attention over the predicted video and tactile tokens. They therefore
  carry forecast information too. Swapping only the tactile slice would not
  remove the forecast's influence. Per plan §4.2, any effect would need matched
  controls before being attributed to tactile prediction. Examples:
  * a refresh that is identical except for the swapped token range;
  * a K/V perturbation of matched norm on the action slice;
  * a refresh of the visual-only context.
* **The forecast is latent.** It is a VAE latent of a tactile *image*, which can
  be decoded through the frozen Wan VAE. Forecast error is therefore measurable
  in image space, but only against a real tactile sensor or a simulated one.

None of this can be exercised without a checkpoint and data (R4, R5).

## 4. Compute available to this project

Checked on 2026-11-18:

* Hardware: one Apple M5 laptop (10 cores, 16 GB unified memory).
* No CUDA, no Docker or Apptainer, no configured remote GPU host.

Estimated requirements. These come from the repositories' own configs; we did
not measure them.

| Route | Requirement | Feasible here? |
|---|---|---|
| TacPAC inference | Backbone: Wan2.2-TI2V-5B (DiT hidden 3072, 30 layers) plus action and tactile experts (hidden 1024, 30 layers each). bf16 weights alone are about 10 GB or more. Needs flash-attn, so CUDA. The paper reports 30.4 ms per correction on its own GPU. | No (CUDA) |
| TacPAC training | Default `NPROC_PER_NODE=8` for both stages, plus our own visuotactile demonstrations. | No |
| T-Rex | Qwen3-VL-2B MoT, CUDA 12.4, real Vega-1 hardware | No |
| ManiFeel | Isaac Gym (TacSL build), Linux, NVIDIA GPU. The repo's example job uses 1 A30 GPU, 24 GB RAM, and an 8-hour limit per diffusion-policy training run. | No locally. **Yes with one rented GPU.** |

## 5. Decision

**TacPAC is the best conceptual fit, and it is selected as the target
architecture.** It is the only released code whose policy combines all three:

* a cached predicted-contact representation;
* live tactile feedback;
* a correction step that can be split at a known token boundary (§3.1).

The literature review identifies it as the most relevant interface for this
question (§1.3).

**But it cannot be reproduced now.** Checkpoints and data are unreleased, it is
real-robot only, and it needs multiple CUDA GPUs. Therefore:

* **No reproduction is claimed.** No TacPAC behaviour, published or adapted, has
  been run.
* **No H2 protocol was registered** and **no H2 experiment was run.**
* **No heuristic controller was substituted.**
* H1 results and artifacts are untouched. The branch is cut from `main` at
  `d490470`.

Rejected as the H2 *subject*:

* **T-Rex and RDP:** no forecast pathway. They remain the reactive baseline the
  plan (§4.3) asks for.
* **VT-WAM, Tactile-WAM, Dream-Tac, ForeTac, ForeTac-VLA, TouchWorld, PredTac,
  TacDyn-WAM, OmniVTA, Contact-Grounded Policy, ref. [13]:** no runnable code
  and checkpoint pair.
* **ViTaL:** a steering method, not an in-loop forecast.
* **ManiFeel checkpoints:** runnable, but they do not forecast.

## 6. Options and the smallest credible next step

| # | Option | What it gives H2 | Missing resources | Rough cost | Main risk |
|---|---|---|---|---|---|
| **1 (recommended)** | **ManiFeel + "TacPAC-structured" small policy.** Start from ManiFeel's released data and its existing tactile diffusion-policy recipe (with the third-party checkpoints as the reproduction target). Add an explicit tactile-forecast head whose *cached* prediction feeds a TacPAC-style single-pass corrector, alongside the live GelSight frame. | A learned policy that relies on both pathways, in a published visuotactile simulator. Pathways are separable by design (R3), and forecast error is measurable against simulated GelSight. Disturbances such as slip, object displacement, and friction change can be injected in Isaac Gym. | One Linux NVIDIA GPU (A30/A10/4090 class) | Reproduce one DP checkpoint, then about 3 variants × 5 seeds ≈ 15 runs × ≤ 8 h. Roughly **100–150 GPU-hours** plus evaluation (an estimate scaled from ManiFeel's job script). | It is still *our* policy, not TacPAC. Results show what a TacPAC-*structured* policy does, not TacPAC itself. Must report this as such. |
| 2 | Same TacPAC-structured policy in **our MuJoCo task, made harder**: mid-episode friction drop, slip, external pull. Small model, trains on the M5. | Runs today, with full control of disturbances and paired seeds. | None | A few CPU or GPU hours per variant | Weak external validity, with an idealized force signal rather than tactile images. H1-002 showed this plant is simple enough that analytic models match learned ones. The disturbance must break that before H2 means anything. |
| 3 | **Wait for the TacPAC release** (or ask its authors for checkpoint access, which is the team's decision to make), then run it on our own visuotactile data | The real model | Checkpoints, a visuotactile sensor on the SO-101, about 8 GPUs to fine-tune, data collection | Large | Unknown release date. Real-robot only. |
| 4 | T-Rex as a reactive H2 *control* (no forecast) | The "faster access to measured touch" comparison the plan requires | Vega-1 hardware or a sim port | Large | Does not test H2 itself |

**Recommendation: Option 1, with Option 2 as a fallback if no GPU can be
obtained.** Do Option 1 in this order:

1. Reproduce one released ManiFeel tactile diffusion-policy checkpoint in its
   own environment, and report the measured success next to the card's number.
2. Add and train the forecast-plus-corrector variant, and confirm it uses the
   forecast under nominal conditions (ablations: forecast zeroed vs. intact).
   Continue to H2 only if it does.
3. Register `H2-001_PROTOCOL.md` with conditions A–E from the brief and the
   matched controls in §3.1. Then run the confirmatory disturbance grid.

Either route builds a policy of our own. Neither is a reproduction of TacPAC,
and every document must say so.
