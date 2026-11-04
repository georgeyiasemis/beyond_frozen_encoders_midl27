# Writing style guide (George Yiasemis, first-author papers 2111.09639, 2512.00872, 2311.15856)

Derived by a reading agent from the arXiv HTML of the three papers (the fourth, the k-space subsampling study, could not be fetched from Semantic Scholar). Essence only; no paper content.

## Abstract
- Order: practical/clinical stake -> limitation ("However, ...", "often ... while ...") -> "In this work we present/introduce X" -> mechanism -> result claim -> release/code sentence.
- Results stated qualitatively in the abstract; metric values belong to the results section. Dataset sizes and factors (e.g. acceleration, number of volumes) may appear.
- No bullets, no headings, no citations in the abstract.

## Introduction and contributions
- Broad opening sentence, "However," for the limitation, "Therefore," / "Motivated by the need to ..." leading to "in this work we propose ...".
- Contributions as a short bullet list, each starting "We propose / introduce / adapt / release / show ...". Priority claims hedged ("At the time of writing, ...").
- Related work critiqued by concession then contrast ("While X ..., this method still ..., which contrasts with our approach ...").

## Sentence level
- "we"/"our" throughout, never "I". Present tense for own claims and formulations, past for what prior work did and for design decisions ("We observed that ... was critical").
- Sentences mostly 20-40 words; short sentences only for emphasis.
- Hedges used: "likely", "may stem from", "could explain", "potential", "further research is needed". Avoided: "prove", "guarantee", "obviously".
- Comparisons: "X outperforms Y", "ranks second or third", gains as "improvements of 5 to 23 percentage points over the next best model".
- Numbers: metrics as decimals (0.724), SDs as (±.001), changes "from X to Y"; counts spelled or numeric but consistent within a paper.
- Punctuation: colons for run-in heads and explanations; parentheses for acronyms and asides; semicolons rare; no em dashes; Latin abbreviations limited to "et al.".
- Inline enumeration "(1) ..., (2) ..."; sequences "First, ... Next, ... Lastly, ...".

## Methods
- Notation declared once ("Throughout the paper, we use ..."); symbols introduced with "Let ... denote ..." and each equation followed by "where ... denotes ...". Bold vectors/matrices, calligraphic operators.
- Numbered equations where a formulation is given; no pseudocode. Configurations inline as tuples. Bold run-in labels for sub-parts.

## Experiments and results
- "Table 3 summarizes ...", "Figure 2 illustrates ..." (spelled out). Baselines introduced with a name, a citation, one or two sentences on what they do, and a critical hedge.
- Ablations narrated as observations; rows labelled "Ablation". Negative results stated plainly ("does not necessarily translate to ...").
- Limitations: name it, hedge the cause, state the next step.
- Metric definitions in prose ("Performance is measured using the macro-averaged Dice ...").

## Terminology
- "pretrained"/"pretraining" (no hyphen), "fine-tuning", "frozen encoder", "encoder" (not "backbone") for the feature extractor, "test set"/"unseen data", "state-of-the-art" hyphenated as adjective, "ground truth" two words.

## Captions
- Figures: "Overview of ..." or a title-like opening followed by one descriptive sentence; panels "(a) ... (b) ...".
- Tables: one definitional sentence, one convention sentence ("Best results are bolded, while second-best results are underlined."), legend with "=" entries.

## Checklist
1 stake -> "However" -> "In this work we"; 2 qualitative abstract results; 3 release sentence; 4 "we", no "I"; 5 "We ..." contribution bullets; 6 hedged priority; 7 concession-contrast for related work; 8 20-40-word sentences; 9 "likely/may stem from"; 10 decimals, percentage points, "from X to Y"; 11 "Let ... denote", "where ... denotes"; 12 "First, Next, Lastly"; 13 ablations as observations; 14 spelled-out Table/Figure refs; 15 caption pattern with bold/underline sentence; 16 "pretrained", "frozen encoder", "test set"; 17 no em dashes, no pseudocode, no "prove".

## Addendum from the fourth paper (arXiv:2301.08365, Magnetic Resonance Imaging 2024)
- Confirms: abstract order gap -> method -> qualitative results -> implication, no metric values; "We ..." contribution bullets after "The contributions and findings of our work can be summarized as follows:"; three-part limitations (name, hedge, next step); "reported in Table ..."; colon-led figure captions ("Experiments pipeline: ...").
- Refines: abstracts may avoid "we" ("This work investigates ...", "Our findings ..."); "In this work, we aim to ..." with a comma; "i.e."/"e.g." do appear in parentheses; spaced hyphens serve as dashes (no em dashes); "Interestingly, ..." and "In summary, our study ..." in the discussion; past tense for own design decisions; "proven ability" appears once, so "prove" is not banned but rare.
- Contradicts: lettered metric definitions (A) SSIM, B) pSNR ...) with equations do appear; only algorithm blocks are absent.
