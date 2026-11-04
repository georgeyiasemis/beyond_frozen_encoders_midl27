# Citation audit: `midl_paper/paper/main.tex` and `refs.bib`

Date: 2026-10-31. Scope: all 54 entries of `paper/refs.bib` (52 cited, 2 unused), every `\cite*` use in `main.tex`, and candidate missing citations. Companion file: `refs_verified.bib` (same keys, fields copied from the primary sources; 19 suggested additions in a separate block at the end).

**Sources used (all public, via `curl`):** arXiv API (`export.arxiv.org/api/query`), Crossref (`api.crossref.org/works`), DataCite (`api.datacite.org/dois`, Zenodo DOI), PMLR volume/paper pages (`proceedings.mlr.press`), CVF open access paper pages (`openaccess.thecvf.com`), OpenReview search endpoint (`api2.openreview.net/notes/search`), `iclr.cc/virtual/2022`, `papers.nips.cc` listings, OpenAlex (`api.openalex.org`, as a second source for venues Crossref does not index), and arXiv HTML/PDF full text for the fit checks of the backbone papers and the closest prior work.
**Sources that could not be used:** Semantic Scholar (HTTP 429 on every call over ~1 h, even at one request per 12 s), DBLP (bot challenge), OpenReview `notes?forum=` endpoint (bot challenge; the search endpoint worked), Springer chapter pages (bot page). Nothing below relies on memory for a bibliographic field; where no reachable source confirmed a field it is marked UNVERIFIED.
**`lit/` folder:** contains notes (`closest_works.md`, `related_work_notes.md`, `peft_variants_recommendation.md`) and an older `refs.bib`, no PDFs. The notes' arXiv IDs were used only as starting hints; every ID was re-fetched.

## 1. Summary

| | count |
|---|---|
| bib entries | 54 (52 cited in `main.tex`; `chen2024cae`, `yang2020covidct` unused) |
| existence verified against a primary source | 54 / 54 |
| entries correct as they stand (at most cosmetic) | 25 |
| entries needing a venue/identifier change (preprint cited although a published version exists, or wrong venue type) | 11: `claessens2025spectre`, `wu2024vocolarge`, `he2024vista3d`, `oquab2023dinov2`, `hu2022lora`, `bardes2024vjepa`, `kim2023ctan`, `ji2022amos`, `rudie2024ratic`, `bilic2023lits`, `chen2024cae` |
| entries needing only added pages/DOI/URL/issue | 18 (see table) |
| uses to REMOVE | `yang2020covidct` (unused; the draft uses COVID-CTset = `rahimzadeh2021covidctset`); one of the `silvarodriguez2023fseft`/`silvarodriguez2025fseft` pair (same paper, workshop + journal) |
| uses to MOVE / reword | `pai2025ctfm` at L51 (CT-FM fine-tunes end-to-end, not "fine-tuned heads"); `claessens2025spectre` at L51 (SPECTRE's frozen protocol is kNN, not a linear/attentive head); `lee2024rsnacsf` at L83 (dataset paper is Lin et al. 2023); `chen2024cae` from unused to L74/L235 next to V-JEPA |
| fit UNVERIFIED | `huang2026mosaic` (no abstract reachable); wording details for `liu2026hyperct`, `zhu2025fmct`, `ly2026seapeft` (see section 3) |
| self-citations | 1 (`veenboer2025tapct`; disclosed in Limitations) |
| verified missing citations | 10 recommended + 7 optional (section 5), including DeLong 1988 and Benjamini-Hochberg 1995 for the statistics at L87 |

## 2. Per-entry table

Verdicts: **OK** = all fields match the source; **FIX** = change the listed fields; **UNUSED** = not cited in `main.tex`. "Evidence" names the record that was fetched on 2026-10-31.

| key | verdict | what to fix | evidence |
|---|---|---|---|
| `veenboer2025tapct` | OK (self-citation) | Fields match PMLR 315:726-753 (2026). Optional: add `url`; key year (2025) differs from `year` (2026). Author "van Veldhuizen" and "Groot Lipman" as on PMLR. | proceedings.mlr.press/v315/veenboer26a.html; arXiv 2512.00872 |
| `claessens2025spectre` | FIX (published) | Now CVPR 2026, pp. 13636-13647: change to `@inproceedings`, `booktitle` CVPR, `pages`, `year=2026`; keep arXiv id in `note`. | openaccess.thecvf.com CVPR2026 paper page (BibTeX block); arXiv 2511.17209v2 |
| `pai2025ctfm` | OK | arXiv 2501.09001 (8 authors match). No journal version found. | arXiv API; Crossref/OpenAlex title searches |
| `blankemeier2026merlin` | OK (minor) | Add `number=8112`; journal title is sentence case; author list puts the last author (Chaudhari, 40th) before "and others" (see section 4). | Crossref 10.1038/s41586-026-10181-8 |
| `wu2024voco` | FIX (minor) | Add `pages=22873--22882`, `doi=10.1109/CVPR52733.2024.02158`. | Crossref 10.1109/cvpr52733.2024.02158 |
| `wu2024vocolarge` | FIX (published) | Published in IEEE TPAMI 48(3):3801-3818, March 2026, DOI 10.1109/TPAMI.2025.3639593: change to journal, `year=2026`. Consider key `wu2026vocolarge`. | Crossref 10.1109/tpami.2025.3639593 |
| `li2024suprem` | OK (minor) | ICLR 2024 (oral) confirmed; add `url` (OpenReview AhizIPytk4). | OpenReview search API; arXiv 2501.11253 comment |
| `he2024vista3d` | FIX (published) | CVPR 2025, pp. 20863-20873, DOI 10.1109/CVPR52734.2025.01943; `year=2025`. Consider key `he2025vista3d`. | Crossref + CVF CVPR2025 paper page |
| `hamamci2026ctrate` | OK (minor) | Add `number=8`; author list pattern (Menze is the 30th author). | Crossref 10.1038/s41551-025-01599-y |
| `dancette2025curia` | OK | arXiv 2509.06830, 23 authors; no published version (Curia-2 at MICCAI 2026 is a different paper). Author list pattern (Hérent is 23rd). | arXiv API; Crossref/OpenAlex |
| `agrawal2025pillar0` | OK | 14 authors match arXiv 2511.17803. A Research Square preprint also exists (10.21203/rs.3.rs-8196619/v1). | arXiv API; OpenAlex |
| `zhu2025fmct` | OK (minor) | Nat Biomed Eng 2026, DOI confirmed; volume/pages not yet assigned (correctly omitted). Journal names the model FM-HCT (as in L51). Key year 2025 vs `year=2026`. | Crossref 10.1038/s41551-026-01668-w (abstract); arXiv 2502.02779v3 |
| `liu2025anymc3d` | OK (minor) | CVPR 2026 pp. 30021-30031 confirmed. Key year 2025 vs `year=2026`. | CVF CVPR2026 paper page; arXiv 2512.12887v3 |
| `baharoon2023dinov2med` | OK | arXiv 2312.02366 (v4, 2024); no published version found. | arXiv API; Crossref/OpenAlex |
| `oquab2023dinov2` | FIX (published) | Published in TMLR 2024: `journal={Transactions on Machine Learning Research}`, `year=2024`, `url` OpenReview a68SUt6zFt. Author list pattern (Bojanowski is 26th). | OpenReview search API ("Accepted by TMLR") |
| `dutt2024peft` | OK (minor) | PMLR 250:406-425 confirmed; add `url`. | proceedings.mlr.press/v250/dutt24a.html |
| `zhu2024melo` | FIX (minor) | Add `pages=1--5`, `doi=10.1109/ISBI56570.2024.10635615`. | Crossref 10.1109/isbi56570.2024.10635615 |
| `hu2022lora` | FIX (venue) | ICLR 2022 confirmed: change to `@inproceedings`, ICLR, `year=2022` (currently arXiv, 2021; key says 2022). | iclr.cc/virtual/2022/poster/6319 (title + authors); arXiv 2106.09685 |
| `liu2024dora` | FIX (minor) | Add PMLR `series`, `volume=235`, `pages=32100--32121`, `url`. | proceedings.mlr.press/v235/liu24bn.html |
| `kumar2022lpft` | OK (minor) | ICLR 2022 confirmed; add arXiv id/URL. | iclr.cc/virtual/2022/papers.html; arXiv comment "ICLR (Oral) 2022" |
| `bardes2024vjepa` | FIX (published) | Published in TMLR 2024 (OpenReview QaCCuDfBk2): change journal, add `url`. | OpenReview search API |
| `chen2024cae` | FIX + UNUSED | IJCV 132(1):208-223, 2024, DOI 10.1007/s11263-023-01852-4 (add volume/pages/DOI). Not cited; either remove or cite at L74/L235 (V-JEPA credits Chen et al. 2022 for the cross-attention pooling behind the attentive probe). | Crossref 10.1007/s11263-023-01852-4; arXiv 2404.08471 full text |
| `ilse2018abmil` | FIX (minor) | Add PMLR `volume=80`, `pages=2127--2136`, `url`. | proceedings.mlr.press/v80/ilse18a.html |
| `zhao2023mtlreview` | OK | All fields match. | Crossref 10.1016/j.compbiomed.2022.106496 |
| `kim2023ctan` | FIX (published) | Published in MICCAI 2023 Workshops (LNCS), pp. 119-128, DOI 10.1007/978-3-031-47401-9_12: change to `@inproceedings`. | Crossref 10.1007/978-3-031-47401-9_12 |
| `yu2020pcgrad` | OK (minor) | NeurIPS 2020 confirmed; add `volume=33` and the papers.nips.cc URL (no DOI exists). | papers.nips.cc 2020 listing; arXiv comment |
| `hatamizadeh2022unetr` | FIX (minor) | Add `pages=1748--1758`, `doi=10.1109/WACV51458.2022.00181`. | Crossref 10.1109/wacv51458.2022.00181 |
| `isensee2021nnunet` | OK (minor) | Add `number=2`. | Crossref 10.1038/s41592-020-01008-z |
| `maierhein2024metrics` | OK (minor) | Add `number=2`; author list pattern (Jaeger is 73rd). | Crossref 10.1038/s41592-023-02151-z |
| `ji2022amos` | FIX (published) | NeurIPS 35 (2022), pp. 36722-36732, DOI 10.52202/068431-2661; proceedings author order is Ji, Bai, Ge, Yang, Zhu, ... (bib has Yang before Ge, the arXiv order). Change to `@inproceedings`. | Crossref 10.52202/068431-2661; OpenReview WqxBRorEbB ("NeurIPS 2022"); papers.nips.cc 2022 |
| `bilic2023lits` | FIX | Add `volume=84`, `doi=10.1016/j.media.2022.102680`. Author list pattern (109 authors). | Crossref 10.1016/j.media.2022.102680 |
| `heller2023kits21` | OK | arXiv 2307.01984, 45 authors match. No KiTS23 overview paper exists in Crossref/arXiv (only LNCS "Kidney and Kidney Tumor Segmentation" 2024 chapters), so citing KiTS21 + the data repository footnote is the best available option. | arXiv API; Crossref/OpenAlex searches |
| `antonelli2022msd` | OK (minor) | Add `number=1`, `pages=4128` (article number). | Crossref 10.1038/s41467-022-30695-9 |
| `flanders2020rsnaich` | OK | All fields match. | Crossref 10.1148/ryai.2020190211 |
| `lee2024rsnacsf` | OK as entry; MOVE as use | Fields match. It describes the winning algorithms, not the dataset; cite Lin et al. 2023, "The RSNA Cervical Spine Fracture CT Dataset", Radiol AI 5(5):e230034, DOI 10.1148/ryai.230034 for the cohort at L83. | Crossref 10.1148/ryai.230256 and 10.1148/ryai.230034 |
| `rudie2024ratic` | FIX (published) | Published in Radiology: Artificial Intelligence 6(6):e240101, Nov 2024, DOI 10.1148/ryai.240101: change from arXiv to journal. | Crossref 10.1148/ryai.240101; OpenAlex |
| `peeters2026luna25` | OK (minor) | Fields match (8(5):e260179). Author list: Silva is 6th, Jacobs 20th; the bib places Jacobs 6th before "and others". | Crossref 10.1148/ryai.260179 |
| `gatidis2022fdgpetct` | OK (minor) | Add `number=1` (article 601). | Crossref 10.1038/s41597-022-01718-3 |
| `yang2020covidct` | UNUSED, REMOVE | Entry is correct (arXiv 2003.13865) but not cited; the draft's COVID cohort is COVID-CTset (`rahimzadeh2021covidctset`). | arXiv API |
| `kalajdzievski2023rslora` | OK | arXiv 2312.03732; no published version found. | arXiv API; Crossref/OpenAlex/OpenReview |
| `touvron2022three` | OK | ECCV 2022 (LNCS), pp. 497-515, DOI match; publisher in Crossref is "Springer Nature Switzerland". LNCS volume number not in Crossref (keep omitted). | Crossref 10.1007/978-3-031-20053-3_29 |
| `wasserthal2023totalsegmentator` | OK | All fields match. | Crossref 10.1148/ryai.230024 |
| `rahimzadeh2021covidctset` | OK | All fields match. | Crossref 10.1016/j.bspc.2021.102588 |
| `wasserthal2026totalsegdata` | OK | Zenodo dataset, version 3.0.0, issued 2026-09-10, title matches. | DataCite 10.5281/zenodo.22688904 |
| `gong20243dsamadapter` | OK | MedIA 98:103324 (2024). | Crossref 10.1016/j.media.2024.103324 |
| `chen2024masam` | OK | MedIA 98:103310 (2024). | Crossref 10.1016/j.media.2024.103310 |
| `wu2025medsa` | OK | MedIA 102:103547 (2025), 8 authors. | Crossref 10.1016/j.media.2025.103547 |
| `fischer2024prompttuning` | OK | MedIA 91:103024 (2024). | Crossref 10.1016/j.media.2023.103024 |
| `silvarodriguez2025fseft` | OK | MedIA 103:103596 (2025). | Crossref 10.1016/j.media.2025.103596 |
| `silvarodriguez2023fseft` | OK; DUPLICATE content | MICCAI 2023 Workshops, pp. 213-224. Same title/authors as the 2025 journal paper; both are cited together at L53: keep the journal version only unless the workshop version is needed. | Crossref 10.1007/978-3-031-47401-9_21 |
| `kartika2026hcc` | OK | PMLR 315:3245-3269. | proceedings.mlr.press/v315/kartika26a.html |
| `liu2026hyperct` | OK | PMLR 315:1768-1801. | proceedings.mlr.press/v315/liu26c.html |
| `huang2026mosaic` | OK (entry); fit UNVERIFIED | MICCAI 2026 (LNCS), pp. 57-67, one author, DOI match; Crossref print date 2027, online 2026-09-26 (keep 2026). No abstract retrievable. | Crossref 10.1007/978-3-032-38072-2_6 |
| `ly2026seapeft` | OK | MICCAI 2026 (LNCS), pp. 559-569, DOI match; arXiv 2603.05822 spells the title "Self-Auditing". | Crossref 10.1007/978-3-032-38069-2_53; arXiv API |

## 3. Per-sentence fit

Line numbers refer to `main.tex` as of 2026-10-31. Verdicts: CORRECT / FIX ENTRY / MOVE / REMOVE / UNVERIFIED; "CHECK" marks a wording detail that the retrieved text does not settle.

| line | sentence (abridged) | key(s) | verdict | basis |
|---|---|---|---|---|
| 51 | "Pretraining on large CT archives has produced a growing set of 3-D foundation models, among them TAP-CT, SPECTRE, Pillar-0, CT-FM, Merlin, CT-CLIP, Curia, FM-HCT and VoCo." | `veenboer2025tapct`, `claessens2025spectre`, `agrawal2025pillar0`, `pai2025ctfm`, `blankemeier2026merlin`, `hamamci2026ctrate`, `dancette2025curia`, `zhu2025fmct`, `wu2024voco`, `wu2024vocolarge` | CORRECT | Each abstract introduces the named model (CT-CLIP is introduced in the CT-RATE paper; the journal version of Zhu et al. names the model FM-HCT). |
| 51 | "Their papers mostly report frozen-encoder results with a linear or attentive head" | `veenboer2025tapct` | CORRECT | TAP-CT: "single linear convolution layer on frozen encoder embeddings" (segmentation) and an ABMIL head on frozen [CLS] embeddings (classification). |
| 51 | same | `claessens2025spectre` | MOVE / reword | SPECTRE's frozen evaluation is a kNN classifier on frozen embeddings ("frozen encoder embeddings are extracted and used to train a k-nearest neighbor classifier without finetuning", protocol of Pai et al.); the text has no linear or attentive head, and the abstract reports "zero-shot and fine-tuned settings". Say "frozen-encoder results with a kNN, linear or attentive head", or cite SPECTRE for the fine-tuned setting. |
| 51 | "some next to fine-tuned heads" | `pai2025ctfm` | MOVE | CT-FM "was adapted for each segmentation and classification task by fine-tuning end-to-end (i.e., adapting all layers)" plus zero-shot embedding tasks; it is an example of full fine-tuning, not of a fine-tuned head on a frozen encoder. Attach it to "or full fine-tuning". |
| 51 | "or full fine-tuning, which probing approached for FM-HCT" | `zhu2025fmct` | CORRECT (CHECK magnitude) | The paper uses linear probing, attentive probing and end-to-end fine-tuning and discusses "the performance difference between linear probing and fine-tuning"; whether probing "approached" fine-tuning was not checked against their tables. |
| 51 | "adapt it with a low-rank update" | `hu2022lora` | CORRECT | LoRA abstract. |
| 53 | "On 2-D medical classification, LoRA matches or beats full fine-tuning of ImageNet-pretrained transformers" | `dutt2024peft`, `zhu2024melo` | CORRECT | Dutt et al.: 17 PEFT methods, ImageNet ViTs, 2-D medical classification (PMLR abstract); MeLo: "Low-rank Adaptation is Better than Fine-tuning". |
| 53 | "probing and fine-tuning of DINOv2 have been compared on radiology tasks" | `oquab2023dinov2`, `baharoon2023dinov2med` | CORRECT | Baharoon et al. sections: linear probing, kNN, few-shot, fine-tuning, PEFT, segmentation. |
| 53, 127 | "found that frozen probing understates both 2-D and native 3-D foundation models ... on RSNA 2023 spleen injury ... full fine-tuning of a convolutional and a Swin 3-D model beat LoRA, which beat linear probing (single task, in their appendix)" | `liu2025anymc3d` | CORRECT (CHECK one clause) | Appendix C2 (arXiv v3): on T4 (spleen injury, RATIC) linear probing 0.654/0.702, LoRA 0.889/0.838, full fine-tuning 0.899/0.919 AUC for MedicalNet (3D ResNet-50) and VoCo (Swin-UNETR); main text: "linear probing ... is inadequate". The clause "LoRA improved 2-D models over a frozen, attention-pooled baseline" is compared in their Tab. 2 against linear probing of MII/MedGemma; confirm "attention-pooled" against their ablation F1 before keeping it. |
| 53 | "adapters that lift the 2-D SAM to volumes" | `gong20243dsamadapter`, `chen2024masam`, `wu2025medsa` | CORRECT | Abstracts: spatial adapters transferring SAM 2D to 3D; 3D adapters in SAM's encoder; Space-Depth Transpose to adapt 2D SAM to 3D. SAM itself is uncited (section 5). |
| 53 | "prompt tuning of a self-supervised 3-D U-Net" | `fischer2024prompttuning` | CORRECT | Prompt-able UNet, frozen after self-supervised pretraining, class-dependent prompt tokens. |
| 53 | "LoRA, adapters and full fine-tuning of one supervised 3-D CT model in few-shot transfer" | `silvarodriguez2023fseft`, `silvarodriguez2025fseft` | CORRECT; REMOVE one of the pair | FSEFT: PEFT and black-box adapters vs full fine-tuning of a CT organ-segmentation foundation model, few-shot. Same paper twice (workshop + journal). |
| 53 | same | `ly2026seapeft` | CORRECT (CHECK wording) | SEA-PEFT automates adapter configuration for few-shot 3-D segmentation; the abstract does not say which backbone, so "one supervised 3-D CT model" needs the paper. |
| 53 | "for a 3-D CT vision-language model, full fine-tuning and PEFT were close on one classification task" | `kartika2026hcc` | CORRECT | PMLR abstract: PEFT AUC 0.94 vs full fine-tuning 0.95 on HCC classification. |
| 53 | "alternatives to plain LoRA that reparameterize or rescale the update" | `liu2024dora`, `kalajdzievski2023rslora` | CORRECT | DoRA: magnitude/direction decomposition; rsLoRA: scale by sqrt(r). |
| 53, 74 | "probe before fine-tuning", "proposed for robustness under distribution shift" | `kumar2022lpft` | CORRECT | LP-FT abstract. |
| 53, 74 | "fine-tune only the attention weights", "trains the query/key/value and output projections in full" | `touvron2022three` | CORRECT | Abstract: "Fine-tuning the weights of the attention layers is sufficient to adapt vision transformers". |
| 53 | "TAP-CT is evaluated for segmentation with a linear convolution on frozen tokens" | `veenboer2025tapct` | CORRECT | See above. |
| 53 | "nnU-Net and models pretrained on segmentation labels remain the references" | `isensee2021nnunet`, `li2024suprem`, `he2024vista3d` | CORRECT | SuPreM: supervised pretraining on AbdomenAtlas 1.1; VISTA3D: segmentation foundation model, 127 classes. |
| 53 | "joint training on many labels can help" | `kim2023ctan` | CORRECT | CTAN improves multi-task learning on four medical datasets (abstract; the +4.67% figure in `lit/` notes is from the paper body). |
| 53 | "or hurt through conflicting gradients" | `yu2020pcgrad` | CORRECT | PCGrad abstract. |
| 53 | "(see ... for a review)" | `zhao2023mtlreview` | CORRECT | Review title. |
| 53 | "joint LoRA suffers from interference between tasks on mixed 2-D and 3-D benchmarks" | `huang2026mosaic` | UNVERIFIED | Entry exists (Crossref) but no abstract was retrievable; the title (mixture-of-specialists adapters, heterogeneous datasets) is consistent with the claim. |
| 53 | "task-specific LoRA weights generated by a hypernetwork approach single-task models on chest CT" | `liu2026hyperct` | CORRECT (CHECK "approach single-task") | Abstract: hypernetwork regresses task-specific LoRA updates on chest CT and "outperforms various strong baselines"; the comparison with single-task models is not in the abstract. |
| 53 | "the closest CT foundation-model studies do not compare joint and per-task adaptation of a 3-D encoder on the same labels" | `veenboer2025tapct`, `liu2025anymc3d` | CORRECT | TAP-CT: one frozen ABMIL head per task; AnyMC3D: per-task plugins on a frozen backbone; neither contrasts joint vs per-task training. |
| 72 | "TAP-CT-B-3D is a 12-block ViT-B (768-d) with 4x8x8 voxel patches, a class token and four register tokens" | `veenboer2025tapct` | CORRECT | Patch size (4, 8, 8) for the 3D configuration; "ViT Register Tokens 4" in the hyperparameter table. |
| 72 | "SPECTRE-Large is a 24-block ViT-L (1080-d) with 16x16x8 patches on 128x128x64 crops ... 64 crops of a 512x512x256 volume at 0.5x0.5x1 mm ... 4-block feature combiner" | `claessens2025spectre` | CORRECT in part (CHECK) | 1080-d, 16x16x8 patches, 128x128x64 crops at 0.5x0.5x1.0 mm and the local/global two-transformer design are in the arXiv v2 text; "24-block", "64 crops of 512x512x256" and "4-block" are not (the text mentions G = 36 windows for report alignment). These presumably come from the released checkpoint config; say so or cite the model card. |
| 72 | "Pillar-0 is used through its head-CT encoder, a convolutional stem and a three-stage multi-scale transformer" | `agrawal2025pillar0` | CORRECT (CHECK "convolutional stem") | Atlas multi-scale attention architecture and a head-CT model are in the paper; the stem is not described in the retrieved text. |
| 72 | "VoCo-L is the encoder of a SwinUNETR-v2 ... released checkpoint trained by self-supervision only, not the variants pretrained with segmentation pseudo-labels" | `wu2024voco`, `wu2024vocolarge` | CORRECT | TPAMI/arXiv text: SwinUNETR backbone, Swin-L feature size 96, "omni-supervised" variants. SwinUNETR itself is uncited (section 5). |
| 74, 235 | attentive probe "in which a learned query token attends over ... tokens through two transformer blocks" | `bardes2024vjepa` | CORRECT | V-JEPA Sec. 4.3 "Evaluation: Attentive Probing" (learnable query, cross-attention pooling on a frozen backbone). V-JEPA credits Chen et al. 2022 (CAE) for this pooling, so `chen2024cae` fits here. |
| 74, 235 | "pooled by gated attention", ABMIL formula | `ilse2018abmil` | CORRECT | ABMIL (gated attention variant is in the paper). |
| 74, 237 | LoRA on q/k/v, rank 16 | `hu2022lora` | CORRECT | |
| 74, 237 | DoRA magnitude per output feature; rsLoRA alpha/sqrt(r) | `liu2024dora`, `kalajdzievski2023rslora` | CORRECT | |
| 76 | "a light multi-depth decoder in the spirit of UNETR" | `hatamizadeh2022unetr` | CORRECT | UNETR: ViT encoder features from several depths into a convolutional decoder. |
| 83 | AMOS (classification cohort and 15-organ segmentation) | `ji2022amos` | CORRECT | |
| 83 | COVID-CTset | `rahimzadeh2021covidctset` | CORRECT | |
| 83 | LUNA25 nodule malignancy | `peeters2026luna25` | CORRECT | |
| 83 | FDG-PET/CT "mutually exclusive diagnosis groups (lung cancer, lymphoma, melanoma, negative)" | `gatidis2022fdgpetct` | CORRECT | Abstract: lymphoma, melanoma, NSCLC and negative controls. |
| 83 | RSNA 2022 cervical-spine fractures | `lee2024rsnacsf` | MOVE | This is the winning-algorithms paper; the dataset paper is Lin et al. 2023 (10.1148/ryai.230034). Cite Lin for the cohort; keep Lee only if the challenge outcome is meant. |
| 83 | RSNA 2023 abdominal trauma | `rudie2024ratic` | CORRECT; FIX ENTRY | Dataset paper (journal version exists). |
| 83 | RSNA 2019 intracranial hemorrhage | `flanders2020rsnaich` | CORRECT | |
| 83 | KiTS23 (footnote to the repository) | `heller2023kits21` | CORRECT with caveat | No KiTS23 overview paper exists; KiTS21 + repository is the best available reference. |
| 83 | LiTS, MSD Pancreas | `bilic2023lits`, `antonelli2022msd` | CORRECT | |
| 83 | TotalSegmentator "dataset version 3.0.0, 1939 CTs ... 117 annotated structures" | `wasserthal2023totalsegmentator`, `wasserthal2026totalsegdata` | CORRECT | DataCite title: "117 important anatomical structures in 1939 CT images", version 3.0.0. |
| 87 | "AUROC and Dice per structure and case are the metrics" | `maierhein2024metrics` | CORRECT | Metrics Reloaded recommendations. DeLong's test and Benjamini-Hochberg in the same paragraph are uncited (section 5). |
| 425 | "the 48 classes of the TAP-CT protocol (left/right pairs, ribs and vertebrae already merged) map to exactly one structure each" | `veenboer2025tapct` | CHECK | The TAP-CT text says merging "result[s] in 49 distinct classes (Appendix C)". If 49 includes background the draft's 48 is consistent (the caption says background is not a class); otherwise correct the number. |

## 4. Bib hygiene

- **Self-citation:** `veenboer2025tapct` (co-author G. Yiasemis). Disclosed at L179 ("One author co-developed TAP-CT"); MIDL 2027 is single-blind per the preamble comment, so no further action.
- **Duplicates:** `silvarodriguez2023fseft` and `silvarodriguez2025fseft` are the workshop and journal versions of one paper, cited together at L53. `wu2024voco` / `wu2024vocolarge` are distinct papers (CVPR 2024 and its TPAMI extension); fine.
- **Preprints with published versions (update):** `claessens2025spectre` (CVPR 2026), `wu2024vocolarge` (TPAMI 2026), `he2024vista3d` (CVPR 2025), `oquab2023dinov2` (TMLR 2024), `hu2022lora` (ICLR 2022), `bardes2024vjepa` (TMLR 2024), `kim2023ctan` (MICCAI 2023 Workshops), `ji2022amos` (NeurIPS 2022), `rudie2024ratic` (Radiol AI 2024), `chen2024cae` (IJCV 2024); `bilic2023lits` lacks volume and DOI. All corrected in `refs_verified.bib`.
- **Still preprint-only (confirmed, keep as arXiv):** `pai2025ctfm`, `dancette2025curia`, `agrawal2025pillar0`, `baharoon2023dinov2med`, `heller2023kits21`, `kalajdzievski2023rslora`, `yang2020covidct`.
- **Key naming:** consistent `surnameYEARtag` scheme, but six keys carry a year different from the entry's `year` after the updates (`veenboer2025tapct`->2026, `zhu2025fmct`->2026, `liu2025anymc3d`->2026, `hu2022lora`->2022 (now consistent), `he2024vista3d`->2025, `wu2024vocolarge`->2026). `refs_verified.bib` keeps the old keys so `main.tex` compiles; rename only if the draft is updated at the same time.
- **Author lists with "and others":** in 11 entries (`blankemeier2026merlin`, `hamamci2026ctrate`, `dancette2025curia`, `oquab2023dinov2`, `maierhein2024metrics`, `ji2022amos`, `bilic2023lits`, `antonelli2022msd`, `lee2024rsnacsf`, `rudie2024ratic`, `peeters2026luna25`) the senior/last author is listed after the first five, before "and others", which misrepresents the order (e.g. Chaudhari is 40th of 40, Jaeger 73rd of 73, Jacobs 20th). `refs_verified.bib` lists the first six in source order; expand to full lists if MIDL requires them.
- **Title capitalisation:** Title Case for most entries but sentence case for the Nature/Elsevier/Zenodo records (`hamamci2026ctrate`, `zhu2025fmct`, `isensee2021nnunet`, `maierhein2024metrics`, `gatidis2022fdgpetct`, `fischer2024prompttuning`, `gong20243dsamadapter`, `chen2024masam`, `wu2025medsa`, `zhao2023mtlreview`, `rahimzadeh2021covidctset`, `wasserthal2026totalsegdata`), and `blankemeier2026merlin` is Title Case although Nature prints sentence case. Acronyms are brace-protected throughout (3D, CT, LoRA, DoRA, SAM, ...), so either convention renders correctly with `jmlr.bst`; pick one.
- **Missing fields fixed in `refs_verified.bib`:** pages/DOI for `wu2024voco`, `zhu2024melo`, `hatamizadeh2022unetr`; PMLR volume/pages for `liu2024dora`, `ilse2018abmil`; issue numbers for `blankemeier2026merlin`, `hamamci2026ctrate`, `isensee2021nnunet`, `maierhein2024metrics`, `antonelli2022msd`, `gatidis2022fdgpetct`; URLs for `li2024suprem`, `dutt2024peft`, `yu2020pcgrad`, `kartika2026hcc`, `liu2026hyperct`, `veenboer2025tapct`.
- **Booktitle style:** mixed ("International Conference on Machine Learning (ICML)" vs full PMLR form). `refs_verified.bib` uses the PMLR form where PMLR is the source and "Proceedings of the IEEE/CVF ..." for CVF venues.
- **Unused entries:** `chen2024cae` (keep and cite at L74/L235, or drop), `yang2020covidct` (drop).

## 5. Missing related work (all verified; identifiers given)

Recommended:

| suggested key | paper | identifier (verified) | sentence it would support |
|---|---|---|---|
| `delong1988` | DeLong, DeLong, Clarke-Pearson, "Comparing the Areas under Two or More Correlated ROC Curves: A Nonparametric Approach", Biometrics 44(3), 1988 | Crossref 10.2307/2531595 | L87 "with DeLong's test on AUROC" |
| `benjamini1995fdr` | Benjamini, Hochberg, "Controlling the False Discovery Rate ...", JRSS-B 57(1):289-300, 1995 | Crossref 10.1111/j.2517-6161.1995.tb02031.x | L87 "with Benjamini--Hochberg correction within each family" |
| `lin2023rsnacsfdata` | Lin et al., "The RSNA Cervical Spine Fracture CT Dataset", Radiol Artif Intell 5(5):e230034, 2023 | Crossref 10.1148/ryai.230034 | L83 "RSNA 2022 cervical-spine fractures" (in place of, or next to, `lee2024rsnacsf`) |
| `kirillov2023sam` | Kirillov et al., "Segment Anything", ICCV 2023, pp. 3992-4003 | Crossref 10.1109/ICCV51070.2023.00371 | L53 "adapters that lift the 2-D SAM to volumes" |
| `hatamizadeh2022swinunetr` | Hatamizadeh et al., "Swin UNETR ...", BrainLes 2021 (LNCS), pp. 272-284 | Crossref 10.1007/978-3-031-08999-2_22; arXiv 2201.01266 | L72 "VoCo-L is the encoder of a SwinUNETR-v2" |
| `he2023swinunetrv2` | He et al., "SwinUNETR-V2: Stronger Swin Transformers with Stagewise Convolutions ...", MICCAI 2023 (LNCS), pp. 416-426 | Crossref 10.1007/978-3-031-43901-8_40 (record found by title search; fetch it for the full author list) | L72 same sentence |
| `liu2021swin` | Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021, pp. 9992-10002 | Crossref 10.1109/ICCV48922.2021.00986 | L72 "four stages of two shifted-window blocks" |
| `dosovitskiy2021vit` | Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021 | arXiv 2010.11929; OpenAlex conference record (ICLR 2021) | L72 "a 12-block ViT-B ... a 24-block ViT-L" |
| `darcet2024registers` | Darcet, Oquab, Mairal, Bojanowski, "Vision Transformers Need Registers", ICLR 2024 (oral) | OpenReview 2dnO3LLiJ1; arXiv 2309.16588 | L72 "a class (CLS) token and four register tokens" |
| `howard2018ulmfit` | Howard, Ruder, "Universal Language Model Fine-tuning for Text Classification", ACL 2018, pp. 328-339 (discriminative per-layer learning rates) | Crossref 10.18653/v1/P18-1031 | L127/L175 "layer-wise learning-rate decay (factor 0.75 per block from the top)" |

Optional (fit a sentence but are not needed for correctness):

| suggested key | paper | identifier | sentence |
|---|---|---|---|
| `standley2020tasks` | Standley et al., "Which Tasks Should Be Learned Together in Multi-task Learning?", ICML 2020, PMLR 119:9120-9132 | proceedings.mlr.press/v119/standley20a.html | L53 "joint training on many labels can help ... or hurt" |
| `fifty2021taskgroupings` | Fifty et al., "Efficiently Identifying Task Groupings for Multi-Task Learning", NeurIPS 2021 | papers.nips.cc 2021 listing; arXiv 2109.04617 | L53 same |
| `hayou2024loraplus` | Hayou, Ghosh, Yu, "LoRA+", ICML 2024, PMLR 235:17783-17806 | proceedings.mlr.press/v235/hayou24a.html | L53 "alternatives to plain LoRA that reparameterize or rescale the update" |
| `lee2023surgical` | Lee et al., "Surgical Fine-Tuning Improves Adaptation to Distribution Shifts", ICLR 2023 | arXiv 2210.11466 (venue from the arXiv comment only) | L51 "train part of it or all of it"; L74 late-block LoRA |
| `bao2022beit` | Bao et al., "BEiT" (layer-wise LR decay for ViT fine-tuning) | arXiv 2106.08254 (ICLR 2022 venue not confirmable via reachable APIs) | L127 layer-wise decay |
| `tang2022swinunetrssl` | Tang et al., "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis", CVPR 2022, pp. 20698-20708 | Crossref 10.1109/CVPR52688.2022.02007 | L51 (earlier 3-D CT self-supervised pretraining) / L72 |
| `ma2024medsam`, `zhou2021modelsgenesis`, `hermans2025rsnatrauma` | MedSAM (Nat Commun 15:654), Models Genesis (MedIA 67:101840), RSNA 2023 trauma challenge review (Radiol AI 7(1):e240334) | Crossref 10.1038/s41467-024-44824-z; 10.1016/j.media.2020.101840; 10.1148/ryai.240334 | L53 context on SAM; L51 history; L83 RSNA 2023 cohort |

Also: `chen2024cae` (already in the bib, unused) belongs at L74/L235 next to `bardes2024vjepa`.

## 6. Not verified

- `huang2026mosaic`: fit of the L53 sentence (abstract not retrievable from Crossref, arXiv or Springer); the entry itself is verified.
- Wording details: `zhu2025fmct` "probing approached" (magnitude), `liu2026hyperct` "approach single-task models", `ly2026seapeft` "one supervised 3-D CT model", `liu2025anymc3d` "frozen, attention-pooled baseline", SPECTRE architecture numbers at L72 (24 blocks, 64 crops of 512x512x256, 4-block combiner), Pillar-0 "convolutional stem", TAP-CT 48 vs 49 classes at L425.
- Venue of `bao2022beit` (ICLR 2022) and `lee2023surgical` (ICLR 2023): only the arXiv author comments state them; OpenReview forum pages were blocked. LNCS volume numbers for all Springer chapters are absent from Crossref and omitted.
- Semantic Scholar returned HTTP 429 throughout; DBLP and the OpenReview forum endpoint served bot challenges; these were replaced by the sources listed at the top.
