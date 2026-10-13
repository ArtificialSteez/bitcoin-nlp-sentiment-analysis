# Bitcoin Tweet Topic Modelling, Sentiment and Volatility

**Using natural language processing to explore online cryptocurrency conversations and test whether sentiment improves next-day Bitcoin volatility forecasts.**

## Overview

This project analyses Bitcoin-related tweets from the **2025–2026** period. It addresses two questions: **What themes appear in Bitcoin discussions?** and **does adding sentiment features improve the prediction of the size of next-day Bitcoin price movements?**

## Methods

- Audit tweet data quality, timestamps, duplicate messages, language filtering and sampling.
- Compare **Latent Dirichlet Allocation (LDA)** on word counts with **Non-negative Matrix Factorisation (NMF)** on TF-IDF vectors.
- Select and interpret a **six-topic NMF** solution and track topic prevalence through time.
- Compare **VADER** and a pretrained **Twitter RoBERTa** sentiment model against **60 manually reviewed tweets**.
- Aggregate daily sentiment and model **next-day absolute Bitcoin log returns** as a volatility proxy.
- Evaluate Ridge and Random Forest forecasting models using chronological holdout and expanding-window validation.

## Main findings

- The six NMF topics separated discussion about trading, NFTs, promotion, rewards and related themes.
- **RoBERTa** matched **44/60** human labels (**73.3%**) versus **38/60** (**63.3%**) for VADER.
- Adding RoBERTa sentiment reduced Ridge's forecast MAE by about **0.78%** relative to its market-plus-activity comparison model, but the improvement was **small and statistically uncertain**.
- The simple **training-mean baseline** achieved the lowest MAE on the final volatility holdout; the evidence does not justify strong trading or causal claims.

## Repository contents

- `EDAB6824_ASS4_Notebook.ipynb` — NLP and volatility modelling workflow.
- `EDAB6808_Assignment4_Report.docx` — written analysis.
- `EDAB6808_Assignment4_Presentation_Lwazi_and_Buhle (1).pptx` — presentation.
- `human_sentiment_labels.csv` — manually reviewed sentiment sample.
- `topic_assignments.csv` — exported tweet-level topic assignments.

## Reproduction

The **full source tweet dataset and several intermediate cached files are not included** in the uploaded ZIP. You will need an authorised copy of the Bitcoin Tweets Dataset 2025–2026 and, where required, cached analysis assets to rerun the workflow from the beginning.

1. Open the notebook in a suitable Python or Colab environment.
2. Follow the notebook's installation and source-recovery cells. Main libraries include pandas, scikit-learn, `pyarrow`, `langdetect`, `vaderSentiment`, `transformers`, `wordcloud` and `yfinance`.
3. Confirm source paths and rerun the analysis in order. Transformer inference may need additional compute and model downloads.

## Limitations

Tweets may contain promotions, duplicates and unverified timestamps. Human sentiment review covers just 60 tweets and only three negative examples. The research is observational, and improvements in sentiment classification need not improve volatility forecasting.

## Contributors

**Lwazi Nhlapo** and **Buhle Mbokodo** — collaborative academic project.
