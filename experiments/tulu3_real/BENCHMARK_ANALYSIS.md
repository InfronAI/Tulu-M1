# Benchmark subgroup analysis

These metrics use the saved 128-token, zero-shot predictions. They remain a reduced protocol.

MATH-500 micro accuracy: 1.40%; equal-weight subject macro accuracy: **0.98%**.

| MATH subject | Correct / count | Accuracy |
|---|---:|---:|
| Algebra | 4/124 | 3.23% |
| Counting & Probability | 0/38 | 0.00% |
| Geometry | 0/41 | 0.00% |
| Intermediate Algebra | 0/97 | 0.00% |
| Number Theory | 0/62 | 0.00% |
| Prealgebra | 3/82 | 3.66% |
| Precalculus | 0/56 | 0.00% |

| MATH difficulty | Correct / count | Accuracy |
|---|---:|---:|
| 1 | 2/43 | 4.65% |
| 2 | 3/90 | 3.33% |
| 3 | 0/105 | 0.00% |
| 4 | 2/128 | 1.56% |
| 5 | 0/134 | 0.00% |

IFEval prompt strict: 14.60%; prompt loose: **17.01%**. Instruction strict: 23.74%; loose: 26.50%.

| Instruction type | Count | Strict | Loose |
|---|---:|---:|---:|
| change_case:capital_word_frequency | 25 | 28.00% | 32.00% |
| change_case:english_capital | 25 | 0.00% | 0.00% |
| change_case:english_lowercase | 39 | 0.00% | 0.00% |
| combination:repeat_prompt | 41 | 29.27% | 29.27% |
| combination:two_responses | 24 | 0.00% | 16.67% |
| detectable_content:number_placeholders | 27 | 22.22% | 22.22% |
| detectable_content:postscript | 26 | 19.23% | 19.23% |
| detectable_format:constrained_response | 10 | 60.00% | 60.00% |
| detectable_format:json_format | 17 | 0.00% | 0.00% |
| detectable_format:multiple_sections | 14 | 14.29% | 14.29% |
| detectable_format:number_bullet_lists | 31 | 12.90% | 12.90% |
| detectable_format:number_highlighted_sections | 48 | 16.67% | 16.67% |
| detectable_format:title | 37 | 48.65% | 48.65% |
| keywords:existence | 39 | 35.90% | 35.90% |
| keywords:forbidden_words | 49 | 32.65% | 44.90% |
| keywords:frequency | 42 | 28.57% | 30.95% |
| keywords:letter_frequency | 33 | 30.30% | 30.30% |
| language:response_language | 31 | 12.90% | 12.90% |
| length_constraints:nth_paragraph_first_word | 12 | 0.00% | 0.00% |
| length_constraints:number_paragraphs | 27 | 7.41% | 11.11% |
| length_constraints:number_sentences | 52 | 44.23% | 50.00% |
| length_constraints:number_words | 52 | 21.15% | 23.08% |
| punctuation:no_comma | 66 | 57.58% | 63.64% |
| startend:end_checker | 26 | 0.00% | 7.69% |
| startend:quotation | 41 | 0.00% | 0.00% |

Stage-panel consistency: 72/72 overlapping questions have identical text and correctness between batch-4 stage evaluation and batch-8 full-bank evaluation. Any differences are retained in the analysis JSON.

Extraction audit: **5** MATH passes use a last-LaTeX candidate appearing verbatim in the question, with neither a boxed nor Minerva final answer. These are flagged for interpretation, not silently rescored. Symbolic equivalence can give credit to repeating an equivalent expression without producing the requested answer; see the saved cases in `benchmark-analysis.json`.

[Tülu 3 §7.2](https://arxiv.org/html/2411.15124v5#S7.SS2) uses MATH subject macro averages and IFEval loose prompt accuracy. Matching the aggregation does not match the model, question subset, prompts or output budget. Instruction checks sharing a prompt are not statistically independent.
