# AI Usage Log

This project used AI-assisted development tools (including code assistants and language tools) as supplementary aids. All experimental design, methodology decisions, data verification, result interpretation, and final review were conducted by team members. AI tools did not generate experimental data, run experiments, or make research decisions.

| Member | Task | AI tool used for | Human review |
|---|---|---|---|
| A (HU Zhongyuan) | Data audit scripts, preprocessing code, sec2/sec3a English prose | Code scaffolding for data loading and EDA scripts; English grammar and phrasing suggestions | Yes — all code executed and outputs verified against raw data; prose reviewed for accuracy of all statistics |
| B (LING Tong) | sick audit scripts, ensemble study code, sec3b English prose | Code structure and debugging assistance (pandas/sklearn syntax); English drafting support for dataset description | Yes — all experimental results independently re-verified against results/ CSV files; medical definitions cross-checked with UCI documentation; every script re-run and output confirmed |
| C (CHEN Yanwu) | Pipeline architecture, config specs, evaluation framework | Code review suggestions; documentation formatting | Yes — pipeline behavior verified by print(pipe) structure checks; statistical tests independently recomputed |
| D (CHEN Yinan) | Evaluation scripts, figures, results chapter | Chart styling suggestions; English phrasing for results interpretation | Yes — all 84 result units independently recomputed from fold-level data; figure values verified against source CSVs |
| E (ZHANG Yijie) | Apriori analysis, report integration, references | Formatting assistance for LaTeX; reference list formatting | Yes — association rules verified by re-running with fixed parameters; final PDF reviewed by all members |

**Nature of AI assistance:**

- **Code development**: AI tools provided syntax suggestions, debugging hints, and boilerplate templates (similar to IDE auto-complete). All code was written, understood, executed, and verified by team members. Every result reported in the paper traces to a specific script in this repository that can be re-run to reproduce the output.
- **English writing**: AI tools assisted with grammar checking and phrasing suggestions for non-native English speakers. The intellectual content — experimental design, analysis decisions, interpretation of results, and conclusions — was determined by team members.
- **What AI did not do**: AI tools did not design the experiment, choose the methods, generate or modify any data, run any experiment, perform any statistical test, or make any research decision. All numbers in the report come from actual experiment outputs stored in `results/`.

**Verification**: Every team member can explain the full pipeline, interpret all reported metrics, and answer questions about any part of the methodology during the presentation.

---

*Last updated: 2026-11-04*
