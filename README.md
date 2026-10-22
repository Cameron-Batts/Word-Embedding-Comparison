# Word Embeddings: Domain-Specific vs. Pretrained Models

**Author:** Cameron Batts  
**Tools:** Python · Gensim · NLTK · pandas · NumPy

## Validation update - October 22, 2026

All 13 default-workflow cells completed on 2,000 reviews, including four ten-epoch embedding fits. Nearest-neighbor examples are kept local.

| Model | Vocabulary | vampire/vampires cosine |
| --- | ---: | ---: |
| w2v_cbow | 11,829 | 0.6598 |
| w2v_sg | 11,829 | 0.5827 |
| ft_cbow | 11,829 | 0.9384 |
| ft_sg | 11,829 | 0.9590 |

The optional Google News download remained disabled. Historical Google News comparisons were not rerun. Multiworker training can vary despite seed 42; cosine similarity is not downstream model accuracy or a cross-space quality ranking.

[Aggregate validation record](reports/validation_results.json). [Repeatable validation script](validation/rerun_analysis.py). Run `python validation/rerun_analysis.py` from the project root. Install `requirements-validation.txt` first; source CSVs stay local.

## Business Problem
Semantic search and content discovery depend on useful text representations. This project explores how domain-specific training, architecture and subword features affect nearest-neighbor meaning.

## Project Approach
- Load 2,000 movie reviews through NLTK.
- Clean and lemmatize text with part-of-speech tags.
- Train Word2Vec and FastText using both CBOW and skip-gram.
- Use 100 dimensions, window=5, min_count=5, epochs=10, seed=42 and workers=4.
- Inspect single-token and combined-token queries.
- Compare selected results with optional 300-dimensional Google News embeddings.

## Historical findings and original context
- Domain Word2Vec skip-gram associated an animation-studio query with animate (0.7660) and animation (0.7316).
- FastText skip-gram associated vampire with vampires (0.9571) but also empire (0.7256), illustrating that subword resemblance can affect results.
- Pretrained computer neighbors included computers and laptop; the domain model returned less relevant terms despite larger cosine scores.
- Similarity scores across different embedding spaces are not a model-quality ranking. No downstream accuracy or business impact was measured.

## Run
```bash
python -m pip install -r requirements.txt
jupyter lab notebooks/word_embedding_comparison.ipynb
```
NLTK downloads the corpus, WordNet and tagger at runtime. Set `RUN_PRETRAINED = True` only when sufficient download bandwidth, disk and RAM are available. Pretrained weights are not bundled. Four FastText/Word2Vec training runs may require substantial memory and time.

## Validation
The default workflow was rerun successfully; earlier outputs remain historical context. Original training emitted repeated Gensim `our_dot_float` numerical warnings; retained results are illustrative and unverified. Multiworker training may vary despite seed=42. Resolve warnings and benchmark retrieval quality before operational use. Tested package versions are recorded in requirements-validation.txt.

## Files
- `notebooks/word_embedding_comparison.ipynb`: workflow and saved validation outputs (source-text displays omitted).
- `reports/word_embedding_comparison.html`: readable analysis.
- `reports/selected_neighbors.csv`: selected non-person neighbors with original ranks and scores.
- `requirements.txt`: dependencies.
- `portfolio_card.html`: project card matching the portfolio.

## Sources
Movie Review Data — Pang and Lee (2004): https://www.cs.cornell.edu/people/pabo/movie-review-data/ . Paper: https://aclanthology.org/P04-1035/ . NLTK distributes the corpus. Gensim provides the optional `word2vec-google-news-300` model through its downloader. Observe each source's applicable terms; raw reviews and pretrained weights are not redistributed here.

## Author
Cameron Batts  
GitHub: https://github.com/Cameron-Batts  
Portfolio: https://cameron-batts.github.io
