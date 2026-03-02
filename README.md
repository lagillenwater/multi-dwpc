# multi-dwpc
This repository contains scripts testing aggregate degree weighted path count (DWPC) statistics. 

# Motivation
Pathway search methods focus on explaining the relationship between two entities, such as a gene and a disease. However, most diseases are the result of more complex interactions between multiple genes and molecular features. For example, many cancers result from mutations across multiple genes. Trisomy 21 is a condition based on the upregulation of the ~225 genes on HSA21, and single-gene searches do not capture the collective mechanistic evidence. Sets of coexpressed or interacting genes thus better capture the cause or effect of disease.  Methods of gene set enrichment do not clearly represent relationships across data modalities that are better captured within a heterogeneous knowledge graph. 

Here we present the scripts for a proof of concept experiment that considers whether the aggregated pathway statistics (mean/median of DWPC) between GO terms and gene annotations differ between the 2015 and 2024 additions. The hypothesis is that the connectivity of hetionet (i.e., the shared connections of genes and pathways to other nodes such as compound, anatomy, and disease) between the 2015 and 2024 GO annotations will result in similar aggregated degree-adjusted pathway scores. Similar scores for the updated GO terms demonstrate the potential of aggregated pathway scores for identifying shared mechanisms of genes not previously annotated to a single shared node (e.g., GO pathway or disease).

# Previous work
Several of the scripts build on previous work from the Greene Lab and [hetionet project](https://het.io/), including the [connectivity-search-analyses](https://github.com/greenelab/connectivity-search-analyses) repository, [hetio/hetnetpy](https://github.com/hetio/hetnetpy), and [hetio/hetmatpy](https://github.com/hetio/hetmatpy). 

# Getting Started

## Clone the repository

```bash
git clone https://github.com/greenelab/multi-dwpc.git
cd multi-dwpc
```

## Create the environment

```bash
conda env create -f env/environment.yml
conda activate multi_dwpc
pip install -e ".[dev]"
```

## Run Pipeline

This stage covers:
- loading and harmonizing 2016/2024 GO-gene data,
- percent-change and IQR filtering,
- optional GO hierarchy analysis for publication mode,
- Jaccard-based redundancy filtering,
- permutation and random null dataset generation.

```bash
# Production pipeline (skips GO hierarchy analysis)
poe pipeline-production

# Publication pipeline (includes GO hierarchy evaluation)
poe pipeline-publication

# Run individual steps
poe load-data
poe filter-change
poe go-hierarchy-analysis
poe filter-jaccard
poe gen-permutation
poe gen-random
```

### Available tasks

Run `poe --help` to see all available tasks:

| Task | Description |
|------|-------------|
| `load-data` | Load Hetionet v1.0 (2016) and GO annotations (2024) |
| `filter-change` | Percent-change + IQR filtering (all_GO_positive_growth) |
| `go-hierarchy-analysis` | GO hierarchy metrics + parents_GO_postive_growth |
| `filter-jaccard` | Jaccard filtering (all_GO, plus parents_GO when available) |
| `gen-permutation` | Generate permutation null datasets |
| `gen-random` | Generate random null datasets |
| `pipeline-production` | Run full production pipeline |
| `pipeline-publication` | Run full publication pipeline |
| `pipeline-null` | Run null dataset generation |

Note: `filter-jaccard` includes parents_GO_postive_growth only when run with
`python scripts/jaccard_similarity_and_filtering.py --include-parents` or via
`poe pipeline-publication`.

### Pipeline scripts

Located in `scripts/`:

1. **load_data.py** - Loads Hetionet v1.0 (2016) and GO annotations (2024), filters to common genes and GO terms
2. **percent_change_and_filtering.py** - Percent-change + IQR filtering for all_GO_positive_growth
3. **go_hierarchy_analysis.py** - GO hierarchy metrics and parents_GO_postive_growth generation
4. **jaccard_similarity_and_filtering.py** - Jaccard filtering for all_GO (and parents_GO when available)
5. **permutation_null_datasets.py** - Generates permutation-based null datasets
6. **random_null_datasets.py** - Generates random null datasets
7. **pipeline_publication.py** - Full publication pipeline runner
8. **pipeline_production.py** - Full production pipeline runner

### Dataset naming

- `all_GO_positive_growth`: all GO terms with positive growth after IQR filtering
- `parents_GO_positive_growth`: parents of leaf terms within the same filtered set

# AI Assistance
This project utilized the AI assistant Claude, developed by Anthropic, during the development process. Its assistance included generating initial code snippets and improving documentation. All AI-generated content was reviewed, tested, and validated by human developers.
