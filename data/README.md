# Dataset directory

Raw datasets are intentionally not committed to GitHub. Download the CWRU
Bearing Data Center files from the official source and store extracted MATLAB
files under data/raw/cwru/.

Suggested local layout:

    data/
      raw/
        cwru/
      processed/

## Dataset notes

- Record the original file name, bearing condition, fault location and size,
  motor load, motor speed (if supplied), channel, and sampling frequency.
- Do not assume every file contains the same channels. Use the
  discover_mat_signals function in src/data_loader.py to inspect a file.
- The loader requires an explicit channel when multiple DE/FE/BA channels exist.
- Keep original files unchanged and preserve source file identifiers in derived data.

CWRU is a seeded bearing-fault dataset. It is useful for an initial
classification experiment, but it is not a run-to-failure CNC spindle dataset
and cannot by itself validate remaining-useful-life predictions for a real CNC.
