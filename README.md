# Customer Feedback Intelligence with NLP

An end-to-end natural language processing pipeline for understanding customer reviews, detecting negative feedback, retrieving similar complaints and supporting confidence-aware human escalation.

**MSc Artificial Intelligence · Natural Language Processing · 2026**  
**Author:** Yolanda N. Nkala · **Module professor:** Dr Abdelaziz Triki

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ahYOxBIZJcmEy3MdFtALpoF9-oqRQGxs?usp=sharing)

## Overview

Customer-support teams need more than a sentiment label: they need to recognise serious complaints, understand recurring themes and know when a model should defer to a person.

This project compares classical TF-IDF classifiers, semantic embeddings and a fine-tuned transformer. Model selection prioritises negative-review recall subject to a precision floor, while calibration, cost-based thresholds and guardrails support more reliable decisions.

## Project files

| File | Contents |
|---|---|
| [Customer_Feedback_NLP.ipynb](Customer_Feedback_NLP.ipynb) | Full workflow, saved outputs, visualisations and interactive Gradio triage demo |
| [Yolanda_Nkala_NLP_Report.docx](Yolanda_Nkala_NLP_Report.docx) | Written analysis, findings, limitations and references |

The notebook and report are the supplied originals, published under shorter filenames.

## Dataset

- **Source:** [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full), introduced by Zhang, Zhao and LeCun (2015).
- **Working sample:** 6,000 reviews drawn from the 650,000-review training split.
- **Labels:** negative = 1–2 stars; neutral = 3 stars; positive = 4–5 stars.
- **Stratified split:** 4,000 training, 800 validation and 1,200 test reviews; seed 42.
- **Prediction input:** review text. Star ratings define the target and risk-analysis groups, not model features.

The full dataset is not included. The notebook downloads it through Hugging Face Datasets.

## Method summary

| Task | Approach |
|---|---|
| Preprocessing and exploration | NLTK and spaCy, negation-aware cleaning, lemmatisation, named entities, collocations and review-length analysis |
| Classical modelling | Staged TF-IDF baselines, five-fold cross-validation, tuning, ablations and probability calibration |
| Semantic search | Word2Vec, GloVe and MiniLM Sentence Transformer embeddings; lexical, semantic and hybrid retrieval; clustering and NMF topics |
| Transformer modelling | DistilBERT fine-tuning, temperature scaling, error analysis and representation comparison |
| Responsible decision support | Privacy redaction, robustness and subgroup checks, confidence-based deferral, safety gates and audit traces |

Vectorisers are fitted within cross-validation pipelines. Calibration and decision thresholds use validation data; final comparisons use the held-out test set.

## Results

Recorded results from the supplied notebook on **1,200 test reviews**:

| Model | Accuracy | Macro F1 | Negative recall | Negative precision |
|---|---:|---:|---:|---:|
| Selected classical model: Complement Naive Bayes | 0.723 | 0.595 | 0.858 | 0.751 |
| Frozen Sentence Transformer + classifier | 0.703 | 0.669 | 0.745 | 0.823 |
| Fine-tuned DistilBERT | **0.792** | **0.739** | **0.879** | **0.846** |

These classification metrics are distinct from the cost-optimised escalation and confidence-based automation results below.

### Key findings

- **Fine-tuning improved sentiment recognition:** DistilBERT's neutral-class recall was 0.479, compared with 0.120 for the selected classical model.
- **Calibration improved probability reliability:** classical expected calibration error fell from 0.124 to 0.042; temperature-scaled DistilBERT achieved 0.031.
- **Decision thresholds reduced urgent misses:** at validation-selected cost thresholds, the classical model missed 3 of 247 one-star reviews and DistilBERT missed 2, with additional false escalations.
- **Human review remained important:** at its validation-selected confidence threshold, DistilBERT automated 67.9% of test reviews at 90.4% accuracy and deferred 32.1% to a person.
- **Governance checks exposed limitations:** both models exceeded the chosen subgroup-gap limit; high confidence did not eliminate prediction errors.

## How to run

**Recommended: Google Colab with a T4 GPU.**

1. [Open the Colab notebook](https://colab.research.google.com/drive/1ahYOxBIZJcmEy3MdFtALpoF9-oqRQGxs?usp=sharing).
2. Select a **T4 GPU** under **Runtime → Change runtime type**.
3. Run the pinned-library installation cell, then choose **Runtime → Restart session**.
4. Continue from the configuration cell and run the remaining cells once, in order.
5. Review the saved tables, figures and the final Gradio triage demo.

The recorded environment uses Python 3.13. Exact versions for the main NLP libraries are specified in the notebook and checked at startup. Dependencies include PyTorch, scikit-learn, NLTK, spaCy, Gensim, Transformers, Datasets, Sentence Transformers, SHAP, WordCloud and Gradio.

Internet access is needed for dataset and pretrained-model downloads. GPU results can vary slightly across hardware and library builds. The Gradio demo uses `share=False`; do not expose real customer text through a public demo.

## Limitations and future work

- Star-derived labels can disagree with the sentiment expressed in ambiguous reviews.
- Results cover one sample and one seed, not a production deployment.
- Demographic fairness could not be measured; observable subgroup checks are not a substitute.
- Retrieval and safety evaluations use small query and challenge sets.
- Costs, precision floors and acceptance criteria are business assumptions.

Future work includes broader evaluation, multilingual and domain-shift testing, improved neutral-class recognition and monitoring of calibration, drift and human-review workload.

## References

Zhang, X., Zhao, J. and LeCun, Y. (2015), *Character-level Convolutional Networks for Text Classification*. Full academic references and methodological discussion are included in the notebook and report.
