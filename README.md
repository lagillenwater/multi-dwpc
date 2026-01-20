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

## Prepare the Data

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
| `compute-dwpc-direct` | Compute DWPC via direct matrix multiplication |
| `test-dwpc-accuracy` | Validate direct DWPC computation against API |
| `benchmark-dwpc` | Benchmark direct vs API computation |
| `pipeline-production` | Run full production pipeline |
| `pipeline-publication` | Run full publication pipeline |
| `pipeline-null` | Run null dataset generation |
| `convert-notebooks` | Convert notebooks to Python scripts |
| `clean` | Remove generated data directories |

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
7. **compute_dwpc_direct.py** - Direct DWPC computation
8. **pipeline_publication.py** - Full publication pipeline runner
9. **pipeline_production.py** - Full production pipeline runner

### Dataset naming

- `all_GO_positive_growth`: all GO terms with positive growth after IQR filtering
- `parents_GO_postive_growth`: parents of leaf terms within the same filtered set

## Run the DWPC computation

There are two methods for computing Degree-Weighted Path Counts (DWPC):

### Option A: Direct computation (recommended)

Computes DWPC directly from the HetMat sparse matrices using hetmatpy. This method is significantly faster and does not require Docker.

```bash
poe compute-dwpc-direct
```

**First run:** Computes and caches all DWPC matrices 

**Subsequent runs:** Loads cached matrices from disk

**Testing accuracy:**

Validate that direct computation matches the API gold-standard values:

```bash
poe test-dwpc-accuracy
```

**Benchmarking:**

Compare direct computation vs API performance:

```bash
poe benchmark-dwpc
```

### Option B: API-based computation

Computes DWPC via the Hetionet API. This requires running the connectivity-search-backend Docker stack.

**1. Start the Docker stack:**

```bash
cd connectivity-search-backend
./run_stack.sh
```

This will:
- Download the postgres database dump (~5 GB) on first run
- Set up the `.env` file with required secrets
- Start the postgres, neo4j, and API containers

The initial database load takes approximately 30 minutes for postgres and 10 minutes for neo4j.

**2. Verify the API is running:**

Wait for the containers to become healthy, then test the API:

```bash
curl http://localhost:8015/v1/nodes/
```

**3. Run the API-based computation:**

```bash
# Uncomment compute-dwpc-api in pyproject.toml first
poe compute-dwpc-api
```

**4. Stop the Docker stack when finished:**

```bash
cd connectivity-search-backend
docker compose down
```

### Performance comparison

Direct computation is 1,000-6,000x faster than API lookups after matrices are cached:

| Method | Time per pair | Notes |
|--------|---------------|-------|
| Direct (cached) | 0.002-0.01 ms | After initial matrix computation |
| API | 12-15 ms | Network overhead per request |

# AI Assistance
This project utilized the AI assistant Claude, developed by Anthropic, during the development process. Its assistance included generating initial code snippets and improving documentation. All AI-generated content was reviewed, tested, and validated by human developers.
