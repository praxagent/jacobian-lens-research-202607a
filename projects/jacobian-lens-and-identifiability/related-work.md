# Related work — brain–LLM alignment and the "workspace layers" lineage

_Context and prior art for our J-space emergence audit. Prompted by a thread from
**Jean-Rémi King** ([@JeanRemiKing](https://x.com/jeanremiking/status/2074500550947680368),
Meta AI / CNRS), which argues that comparing LLM internals to the human brain — and
finding **middle layers special** — is a well-established program that predates and
parallels Anthropic's J-space. Reading his primary sources; this is the neuroscience
half of the story our [`background.md`](background.md) should not omit._

## Why this matters to our project

Two direct connections:

1. **The "middle layers are special" phenomenon has deep prior art.** Our audit measures
   a *mid-network* CKA band (`experiments/jacobian_lens/`). The neuro-AI literature
   independently found, years earlier, that an LLM's **middle layers** align best with
   the human brain and behave like **associative cortex**. Anthropic's own authors
   (Gurnee, Sofroniew) named these "**Workspace Layers**" — but the *locus* (middle
   layers) and much of the *interpretation* (a flexible, integrative, planning-relevant
   stage) was established by King, Caucheteux, Huth, Schrimpf, Goldstein and others.
2. **It sharpens the interrogation, honestly.** King is explicit (post 11): this does
   **not** make Anthropic's work incorrect — it's an "excellent investigation." His
   point is about **credit and framing**: Global Workspace Theory is primarily about
   *perception*, not language, and the brain–LLM comparison community deserves citation.
   This aligns with our audit's actual thesis: the *measurements* are strong; the
   *novelty/framing* of "a global workspace emerged" outruns what's new once you account
   for (a) the prior middle-layer/brain-alignment literature and (b) the unresolved GWT
   criteria (ignition, etc., per Dehaene & Naccache).

## The lineage (King's thread, primary sources)

**LMs spontaneously align with the brain, most in the middle layers.**
- Schrimpf et al. 2021, *PNAS* — integrative modeling; LM predictivity of brain/behavior.
  [pnas.org/10.1073/pnas.2105646118](https://www.pnas.org/doi/10.1073/pnas.2105646118)
- Caucheteux & King 2022, *Communications Biology* — "Brains and algorithms partially
  converge in natural language processing."
  [nature.com/s42003-022-03036-1](https://www.nature.com/articles/s42003-022-03036-1)
- Jain & Huth 2018, *NeurIPS* — context in LM encoding models of fMRI; alignment peaks in
  **associative speech cortex**.
  [proceedings.neurips.cc/…2018…](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f471223d1a1614b58a7dc45c9d01df19-Abstract.html)

**Long-horizon forecast layers align with the fronto-parieto-temporal network.**
- Caucheteux et al. 2023, *Nature Human Behaviour* — "Evidence of a predictive coding
  hierarchy in the human brain listening to speech."
  [nature.com/s41562-022-01516-2](https://www.nature.com/articles/s41562-022-01516-2)

**Layer-wise processing follows the cortical hierarchy.**
- Millet et al. 2022, *NeurIPS* — self-supervised speech models & the cortical hierarchy.
  [arxiv.org/abs/2206.01685](https://arxiv.org/abs/2206.01685)
- Goldstein et al. 2022, *Nature Neuroscience* — "Shared computational principles for
  language processing in humans and deep language models."
  [nature.com/s41593-022-01026-4](https://www.nature.com/articles/s41593-022-01026-4)

**But the alignment is imperfect — a nonlinear mapping does much better.**
- Meta AI / TRIBE (2026) — nonlinear brain↔model mapping.
  [aidemos.atmeta.com/tribev2](https://aidemos.atmeta.com/tribev2/) *(King's caveat: linear
  brain-alignment is a floor, not a ceiling — relevant to how much to read into any linear
  probe/lens, ours included.)*

**Alignment enables decoding language & images from brain activity.**
- Tang et al. 2023, *Nature Neuroscience* — semantic reconstruction from non-invasive
  recordings. [nature.com/s41593-023-01304-9](https://www.nature.com/articles/s41593-023-01304-9)
- Défossez et al. 2023, *Nature Machine Intelligence* — decoding speech from MEG/EEG.
  [nature.com/s42256-023-00714-5](https://www.nature.com/articles/s42256-023-00714-5)
- Meta *Brain2Qwerty* (Zhang, Lévy et al. 2026).
  [facebookresearch.github.io/brain2qwerty](https://facebookresearch.github.io/brain2qwerty/)
- Vision: Yamins et al. 2014, *PNAS* [pnas.org/10.1073/pnas.1403112111](https://www.pnas.org/doi/10.1073/pnas.1403112111);
  brain→image: Ozcelik & VanRullen 2023, *Sci. Reports*
  [nature.com/s41598-023-42891-8](https://www.nature.com/articles/s41598-023-42891-8);
  Scotti et al. *MindEye2* [arxiv.org/abs/2403.11207](https://arxiv.org/abs/2403.11207).

Researchers named in the thread are logged in
[`shared/researchers/handles.yaml`](../../shared/researchers/handles.yaml) for the future
daily research scan.

## Where this points our research (candidate directions)

1. **Bridge our band to brain-alignment (the strongest new idea).** We measure a
   mid-network *workspace band* via CKA of J-lens geometry, across 38 models and scale.
   The neuro-AI community measures *brain-alignment* per layer, also peaking in the
   middle. **Do they coincide?** If, across the same model families, our `mid_sep`
   emergence tracks the emergence of peak brain-alignment (e.g. via Brain-Score /
   published per-layer curves), that's a novel, citable bridge between an
   interpretability statistic and an independent neuroscience one — and it would tell us
   whether "workspace band" and "brain-aligned associative layer" are the same object.
2. **Credit + framing in our writeup.** Any public post must cite this lineage and adopt
   King's honest register: excellent Anthropic work, but the middle-layer-workspace idea
   and the brain comparison are not new — which is itself part of the "was the framing
   oversold?" answer.
3. **Linear-probe humility.** TRIBE shows linear brain↔model maps badly underestimate
   alignment. The J-lens (and our CKA on it) is a *linear* readout; that's a reason to
   treat "the model has a workspace" claims — and our own linear-CKA band — as lower
   bounds on structure, not the whole story. Worth stating plainly.

## Community reception (discourse, not evidence)

The J-space paper drew a spectrum of public reaction worth tracking — not as findings, but
as the discourse our empirical audit sits within:

- **Sympathetic expert — Dehaene & Naccache** (GNW originators): "a landmark," with
  specific caveats (ignition undemonstrated; capacity likely ~6 not ~25). See
  [background.md](background.md).
- **Prior-art credit — Jean-Rémi King** (Meta / CNRS): the "middle layers / workspace"
  result and brain–LLM comparison have a decade-deep lineage the paper under-credits, and
  GWT is primarily about *perception*, not language (this doc, above).
- **Deflationary critique — [@ziv_ravid](https://x.com/ziv_ravid/status/2074581837746176021)**
  (Meta): the core interpretability result is fine and the J-lens is a reasonable tool, but
  the consciousness / GWT / "brain" framing is a *separable packaging choice* that adds
  nothing to the linear algebra — "the choice is the product, not the science." Extends to a
  broader claim that much of Anthropic's output is PR.
- **Technical-deflationary critique — Andrew Trask (@iamtrask, OpenMined; thread
  2026-07-08):** "very nice mech interp research," but the result is *expected*: models
  learn only through gradients so "the gradients will have semantic structure," and
  residual streams ("passthrough gradients") make cross-layer transport unsurprising —
  "it would be more surprising if it didn't work." Plus: "I miss peer review."
  **Our assessment (both directions):** the argument conflates training gradients
  (∂L/∂θ) with the frozen forward function's Jacobian (∂h_final/∂h_l) — a random-init
  network has equally-computed Jacobians with no semantic structure, so structure is a
  property of the *learned function*, not of backprop per se. The residual-stream
  steelman is real for mere decodability (it's why logit lens works) — but **our data
  refutes the "architecture-guarantees-it" form**: Gemma-3-12b (residual + backprop) has
  essentially **no** band (0.0007) while qwen3-14b has a strong one (0.21); small models
  show none; the random-transport null shows none; and Nanda finds J-lens *beats* logit
  lens (information beyond passthrough). His peer-review point, however, lands — and is
  corroborated by Dehaene & Naccache's note that the paper evolved during their
  commentary exchange (no pre-registration).

**Our stance.** The framing critique (Ravid, King) aligns with this project's thesis, and
our data supports it: the mid-band is **family-dependent and instruct-reduced**, not the
universal phenomenon the framing implies. But "it's all PR" is itself a narrative overreach
— the GWT connection is *contestable science* (its originators engaged it seriously), not
mere marketing. And the Trask thread sharpens the symmetric point: **the hype and the
anti-hype make the same mistake — treating the phenomenon as *necessary*** (either "a
workspace like consciousness" or "a trivial consequence of backprop + residual streams")
**when the data shows it is *contingent***: a learned, family-dependent structure that some
training recipes produce strongly (Qwen) and others barely at all (Gemma), absent in small
models and in the null. The useful response to both is **measurement**, which is what this
project brings: not picking a side in the discourse, but publishing the curve. Reaction
handles are logged in
[`shared/researchers/handles.yaml`](../../shared/researchers/handles.yaml).

## Sources

- King thread (2026): [x.com/jeanremiking/status/2074500550947680368](https://x.com/jeanremiking/status/2074500550947680368)
- Ravid critique (2026): [x.com/ziv_ravid/status/2074581837746176021](https://x.com/ziv_ravid/status/2074581837746176021)
- Dehaene & Naccache commentary: [unicog.org PDF](https://unicog.org/wp_2025/wp-content/uploads/2026/07/Dehaene-and-Naccache-Workspace-commentary-on-Gurnee-Lindsey-June-2026.pdf)
- All papers linked inline above (from the King thread).
