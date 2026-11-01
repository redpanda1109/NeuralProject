# Delulu Dataset Exploration Report

Source file: `data/delulu.csv` (copy of the original; the original was not modified).

## 1. Size
- Rows (benchmark items): **1947**
- Columns: **13**
- Each row is a **pair**: one correct (`golden_completion`) and one hallucinated
  (`hallucinated_completion`) completion for the same code context
  (`prompt` = code before the gap, `suffix` = code after the gap).
- After expanding pairs for the binary task: **3894** examples (50% correct / 50% hallucinated).

## 2. Columns, data types and missing values
| column | dtype | missing | missing_% | unique_values |
|---|---|---|---|---|
| benchmark_id | object | 0 | 0.0 | 1947 |
| image_tag | object | 0 | 0.0 | 1947 |
| language | object | 0 | 0.0 | 7 |
| file_path | object | 0 | 0.0 | 507 |
| hallucination_type | object | 0 | 0.0 | 4 |
| prompt | object | 125 | 6.42 | 921 |
| suffix | object | 12 | 0.62 | 1035 |
| golden_completion | object | 0 | 0.0 | 906 |
| hallucinated_completion | object | 0 | 0.0 | 1816 |
| error_message | object | 0 | 0.0 | 1453 |
| metadata | object | 0 | 0.0 | 1947 |
| license | object | 0 | 0.0 | 27 |
| repo_url | object | 3 | 0.15 | 319 |

## 3. Duplicates
- Fully duplicated rows: **0**
- Duplicated `benchmark_id`: **0** (every benchmark_id is unique, so splitting by
  benchmark_id is the same as splitting by original row)
- Duplicated (prompt, suffix, golden, hallucinated): **9**
- Duplicated (prompt, suffix, golden): **867**
- Distinct code contexts (prompt + suffix): **1077**

**Important:** the benchmark often builds 2-3 items (different hallucination types) from the
SAME location in the same file. Those items share prompt, suffix and golden completion, so their
correct examples are identical. A plain row split would place identical correct examples in
both training and test. `data_split.py` therefore keeps each code context inside one split.

## 4. Programming languages (7)
| language | count | percent |
|---|---|---|
| typescript | 420 | 21.57 |
| python | 370 | 19.0 |
| go | 291 | 14.95 |
| rust | 252 | 12.94 |
| csharp | 246 | 12.63 |
| java | 243 | 12.48 |
| cpp | 125 | 6.42 |

## 5. Hallucination categories (Task B labels)
| hallucination_type | count | percent |
|---|---|---|
| undefinedvariable | 576 | 29.58 |
| import | 476 | 24.45 |
| method | 460 | 23.63 |
| parameter | 435 | 22.34 |

### Language x category
| language | import | method | parameter | undefinedvariable |
|---|---|---|---|---|
| cpp | 24 | 32 | 35 | 34 |
| csharp | 26 | 51 | 82 | 87 |
| go | 50 | 81 | 79 | 81 |
| java | 44 | 59 | 65 | 75 |
| python | 164 | 77 | 22 | 107 |
| rust | 57 | 60 | 66 | 69 |
| typescript | 111 | 100 | 86 | 123 |

## 6. Length statistics
| index | 25% | 50% | 75% | 95% | count | max | mean | min | std |
|---|---|---|---|---|---|---|---|---|---|
| prompt (chars) | 236.5 | 974.0 | 2930.0 | 9400.0 | 1947.0 | 22650.0 | 2263.5 | 0.0 | 3209.8 |
| prompt (lines) | n/a | 31.0 | n/a | 221.0 | 1947.0 | 536.0 | 62.2 | 1.0 | 80.2 |
| suffix (chars) | 467.0 | 1349.0 | 3775.0 | 11798.0 | 1947.0 | 32759.0 | 2989.7 | 0.0 | 4192.5 |
| suffix (lines) | n/a | 46.0 | n/a | 334.0 | 1947.0 | 858.0 | 89.6 | 1.0 | 109.3 |
| golden_completion (chars) | 24.0 | 37.0 | 62.0 | 210.0 | 1947.0 | 1089.0 | 66.7 | 5.0 | 96.1 |
| golden_completion (lines) | n/a | 1.0 | n/a | 6.0 | 1947.0 | 33.0 | 2.0 | 1.0 | 3.3 |
| hallucinated_completion (chars) | 31.0 | 46.0 | 72.0 | 229.0 | 1947.0 | 1151.0 | 75.8 | 6.0 | 96.9 |
| hallucinated_completion (lines) | n/a | 1.0 | n/a | 6.0 | 1947.0 | 33.0 | 2.0 | 1.0 | 3.3 |

- Hallucinated completion is longer than the golden one in **88.2%** of pairs
  (a length bias that simple models could exploit).

## 7. Repositories and file paths (leakage analysis)
- Distinct repositories: **319** (missing repo_url: 3)
- Items from the largest repository: **176**
- Distinct file paths: **507**
- Inside a pair, `prompt`, `suffix`, `file_path`, `repo_url` and `license` are **identical** for the
  correct and the hallucinated completion. Therefore `file_path` carries **no information** that can
  separate the two completions; it could only let a model memorise repositories. It is **not** used
  as model input. `hallucination_type`, `error_message`, `benchmark_id`, `image_tag` and `metadata`
  directly reveal the label/category and are also excluded.

## 8. Suspicious / unexpected records
| issue | records |
|---|---|
| very short prompt (< 20 chars) | 146 |
| missing prompt (completion is at the start of the file) | 125 |
| missing suffix (completion is at the end of the file) | 12 |
| missing repo_url | 3 |

Full list: `outputs/metrics/eda_suspicious_records.csv`.
Missing prompts/suffixes are legitimate (the completion is at the very start or end of a file); they
are replaced by an empty string during preprocessing. No records were removed.

## 9. Example records per category
### `import`

**typescript-import-e502457b899f** (typescript)

```
# ...end of prompt...

```
- golden: `import { test, expect } from '@playwright/test';`
- hallucinated: `import { test, expect } from '@playwright/testing';`
- error message (analysis only, never model input): `Cannot find module '@playwright/testing' or its corresponding type declarations.`

**python-import-e33e114103a8** (python)

```
# ...end of prompt...
import numpy as np
import pandas as pd

```
- golden: `import anndata`
- hallucinated: `import anndata2`
- error message (analysis only, never model input): `Package 'anndata2' could not be installed (completion import)`

### `method`

**java-method-574dacbd6e07** (java)

```
# ...end of prompt...
...).toString();

            records.add(new KinesisAnalyticsInputPreprocessingResponse.Record(record.getRecordId(), Ok, ByteBuffer.wrap(reversedString.getBytes(UTF_8))));
        }

        return new 
```
- golden: `KinesisAnalyticsInputPreprocessingResponse(records);`
- hallucinated: `KinesisAnalyticsInputPreprocessingResponse.buildResponse(records);`
- error message (analysis only, never model input): `cannot find symbol`

**rust-method-a4c087c58c50** (rust)

```
# ...end of prompt...
...              original: pattern_str,
                        pattern,
                        negated,
                    });
                }
                Err(e) => {
                    return 
```
- golden: `...rn on line {}: '{}': {}",
                        line_num + 1,
                        pattern_str,
                        e
                    ));`
- hallucinated: `...rn on line {}: '{}': {}",
                        line_num + 1,
                        pattern_str,
                        e
                    ));`
- error message (analysis only, never model input): `cannot find macro `format_error` in this scope`

### `parameter`

**rust-parameter-fe0cfd35546b** (rust)

```
# ...end of prompt...
...       .unwrap_or_else(|| DEFAULT_THEME_ID.to_string())
}

/// Validate a theme and return its label for messaging.
pub fn ensure_theme(theme_id: &str) -> Result<&'static str> {
    REGISTRY
        .
```
- golden: `get(theme_id)
        .map(|definition| definition.label)
        .context("Theme not found")`
- hallucinated: `...inition| definition.label).context(format!("Theme '{}' not found", theme_id), "suggestion = Use `available_themes()` to list valid theme identifiers")`
- error message (analysis only, never model input): `this method takes 1 argument but 2 arguments were supplied`

**csharp-parameter-af023ef3604b** (csharp)

```
# ...end of prompt...
...bscribingAll"),
                "Unsubscribing all {SubscriptionCount} subscriptions");

            _disposingSocketClient = LoggerMessage.Define(
                LogLevel.Debug,
                new 
```
- golden: `EventId(3015, "DisposingSocketClient")`
- hallucinated: `EventId(3015, "DisposingSocketClient", loggerName: "SocketApiClient")`
- error message (analysis only, never model input): `The best overload for 'EventId' does not have a parameter named 'loggerName' [/tmp/verify_cs_gu06q49y/verify.csproj]`

### `undefinedvariable`

**typescript-undefinedvariable-675a1997cdf6** (typescript)

```
# ...end of prompt...
...seIntersectionObserver(
  ref: RefObject<HTMLDivElement | null>,
  options: IntersectionObserverInit = { threshold: 0.1 },
): boolean {
  const [isIntersecting, setIntersecting] = useState(false);

  
```
- golden: `...ns);

    if (ref?.current) {
      observer.observe(ref.current);
    }

    return () => {
      observer.disconnect();
    };
  }, [ref, options]);`
- hallucinated: `...ns);

    if (ref?.current) {
      observer.observe(ref.current);
    }

    return () => {
      observer.disconnect();
    };
  }, [ref, options]);`
- error message (analysis only, never model input): `Cannot find name 'observerOptions'.`

**python-undefinedvariable-301b74b0fb04** (python)

```
# ...end of prompt...
...   )

    op.create_table(
        "optimization_scalar_docs",
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("dimension__id", sa.Integer(), nullable=True),
        sa.
```
- golden: `...      "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        )`
- hallucinated: `...         sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=column_increment),
            nullable=False,
        )`
- error message (analysis only, never model input): `invalid syntax`


## 10. Figures
- `outputs/figures/eda_hallucination_categories.png`
- `outputs/figures/eda_languages.png`
- `outputs/figures/eda_language_by_category.png`
- `outputs/figures/eda_completion_lengths.png`
- `outputs/figures/eda_prompt_suffix_lengths.png`
- `outputs/figures/eda_missing_values.png`
