# IU X-Ray dataset for R2GenGPT

Downloaded on 2026-09-25. Dataset files only; no prototype or model has been built.

## Verified contents

- 6,091 PNG image files in the authors' package.
- 2,955 annotated cases referencing 5,910 unique images; all referenced images are present.
- The additional 181 PNG files are retained exactly as supplied.
- 3,955 original XML reports.
- Approximately 2.1 GiB on disk including the retained download archives.
- ZIP CRC checks, all PNG integrity checks, and all XML parse checks passed.

## Sources

- R2GenGPT authors' official repository: https://github.com/wang-zhanyu/R2GenGPT
- IU X-Ray package linked by that repository: https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view
- Original XML reports from the U.S. National Library of Medicine: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
- Original collection information and usage information: https://openi.nlm.nih.gov/faq

## Folder contents

- `images/`: images from the authors' prepared IU X-Ray package.
- `annotation.json`: the authors' report annotations and train/validation/test split.
- `reports/ecgen-radiology/`: 3,955 original XML radiology reports from NLM Open-i.
- `archives/iu_xray.zip`: original downloaded authors' package.
- `archives/NLMCXR_reports.tgz`: original downloaded XML report archive.
- `verification.json`: measured file counts, integrity checks, and archive SHA-256 hashes.

The prepared annotations contain 2,069 training cases, 296 validation cases, and 590 test cases (2,955 total). Each case contains `image_path` entries relative to `images/` and a report. These existing splits are preserved unchanged.

The prepared package is a curated subset of the original collection described in the study guide. The 3,955 original XML reports cover a broader collection and are kept separately; do not assume that every XML report has an image in the prepared subset. The original 7,470-image NLM archive was not retained because the authors' prepared package is used instead.

MIMIC-CXR has not been downloaded. The authors link to PhysioNet, where it requires credentialed access: https://physionet.org/content/mimic-cxr-jpg/2.0.0/

Consult the original providers for dataset usage terms. The R2GenGPT code repository's license should not be assumed to apply to the medical dataset.
