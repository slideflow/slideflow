# Changelog

All notable changes to Slideflow are documented in this file. See git
history for the full list of commits behind each entry.

## 3.1.0 — 2026-05-XX

A focused stability and bug-fix release on top of 3.0.2, with a small
number of additive features and a few targeted behavior corrections.
Most user-visible behavior is unchanged; items that *do* change
observable output are flagged with ⚠️ below.

This release is numbered `3.1.0` rather than `3.0.3` because it adds
new public surface (extractors, methods, attributes) and includes
several behavior-changing fixes that affect reproducibility of prior
training runs.

### Added

- **H-optimus-0 feature extractor** — new `hoptimus0` extractor
  (1536-dim ViT, 224 px tiles, Apache-2.0 license; weights from
  https://huggingface.co/bioptimus/H-optimus-0).
- **Weights hash for `virchow` and `hoptimus0`** — new
  `BaseFeatureExtractor.weights_hash` class attribute, populated for
  both extractors. Useful as a stable identity tag when loading via
  HuggingFace auto-download.
- **`WSI.roi_area()`** — returns per-ROI areas in mm².
- **`ROI.area()`** — returns single-ROI area; **`ROI.__deepcopy__`**
  added for proper deep-copy semantics (resets cached geometries).
- **`TileMaskDataset.split(train_size, seed)`** — convenience method
  returning `(train_subset, val_subset)` Subsets, split by slide.
- **Studio ROI copy/paste** — Ctrl+C / Ctrl+V keybinds in Slideflow
  Studio for copying ROIs between slides; new Edit option in the ROI
  right-click menu.
- **`location_heatmap()` enhancements** — `filename` is now optional
  (skips plotting if None), accepts a `stride` kwarg, and returns the
  generated `np.ndarray` heatmap.
- **Plugin loader is now defensive** — third-party plugin entry-points
  that fail registration log a warning and are skipped, rather than
  aborting `import slideflow`.

### Changed

- **Segmentation models switched from Adam → AdamW** for training. ⚠️
  Different weight-decay semantics; users re-running existing training
  recipes will see slightly different convergence.
- **DataLoader workers now use distinct NumPy seeds** (`worker_init_fn`
  mixes `worker_id` into the parent's RNG state instead of using it
  directly). ⚠️ Fixes correlated NumPy-driven augmentations across
  workers; reproducibility-sensitive runs that relied on the old
  shared-seed behavior will see different augmentation streams.
- **MIL bundle loading in Studio** is now tolerant of unknown future
  param keys (`mil_config(..., validate=False)` on the Studio load
  path) — Studio can now display models trained on a future Slideflow
  version that adds new `mil_params.json` keys.
- **Image-type detection migrated from `imghdr` to `filetype`** for
  Python 3.13+ compatibility. `filetype` is now a hard install
  dependency.
- **`crc32c` is now a hard import-time dependency** — `import slideflow`
  raises a clear `ImportError` at import time instead of crashing later
  in `tfrecord/writer.py`. `crc32c` was already in `install_requires`.

### Deprecated

- **`Dataset.resize_tfrecords(tile_px)`** — emits `DeprecationWarning`;
  removal scheduled for Slideflow 4.
- **`Dataset.read_tfrecord_by_location(decode=...)`** kwarg — emits
  `DeprecationWarning` if passed; the record is always decoded.
  Removal scheduled for Slideflow 4.

### Removed

- **`slideflow/model/adv_utils.py`** — unused TF-only adversarial-
  training helper (`make_adversarial_model`, `convert_dataset`); zero
  callers in the codebase. Removed without deprecation since no public
  re-export ever existed.

### ⚠️ Behavior changes affecting prior outputs

Users who rely on bit-exact reproducibility of prior training runs
should be aware of the following corrections. Most of these were
silent bugs in 3.0.2; the fix changes the *correct* output.

#### Math / training-objective fixes

- **`batch_loss_crossentropy` regularizer divisor** was dividing
  variance/SE by feature count (post-`reduce_mean` collapse) instead
  of sample count. Magnitude of change depends on `num_features` vs
  batch split size.
- **`softmax_percent` / `softmax_predict` with non-contiguous
  `prediction_filter`** — argmax results were written into wrong class
  slots when `prediction_filter` skipped classes (e.g., `[0, 2]`
  populated slots 0 and 1). Users with class-skipping filters should
  re-run inference.

#### Survival / Cox NLL — convention clarified, no behavior change

Slideflow 3 documentation was unclear about the convention used by
`'negative_log_likelihood'` survival loss. The convention has been
confirmed and documented in-code:

- The default `'negative_log_likelihood'` loss treats the model output
  as a **survival score** (higher value = longer expected survival;
  lifelines / Harrell convention), **not** a Cox log-hazard.
  Mathematically, this requires *ascending* sort by time inside the
  loss — the implementation in 3.0.2 is correct as written. Earlier
  drafts of this changelog incorrectly described the ascending sort
  as a bug; that mischaracterization has been retracted.
- A pre-release commit on the 3.1.0 branch briefly flipped the sort to
  descending under the same misreading. That change has been reverted.
  **Behavior of `'negative_log_likelihood'` in 3.1.0 is identical to
  3.0.2.** Survival models trained on Slideflow 3.0.x do not need to
  be retrained.
- The variant `'negative_log_likelihood_breslow'` does use the
  opposite (Cox log-hazard, descending sort) convention; the two
  losses are not directly interchangeable. Multi-paragraph in-code
  comments in `model/tensorflow_utils.py` and `stats/metrics.py`
  document the convention difference for future contributors.

#### Training-time concordance index displays anti-concordantly

Independent of any behavior change, this 3.1.0 release surfaces a
long-standing quirk that was never documented:

- Slideflow has **two** `concordance_index` implementations with
  **opposite** conventions:
  - `slideflow.stats.metrics.concordance_index` (post-evaluation;
    appears as `patient_c_index`, `slide_c_index`, `tile_c_index` in
    `results_log.csv`) follows the survival-score convention. **This
    is the authoritative metric.**
  - `slideflow.model.tensorflow_utils.concordance_index` (the
    training-time Keras metric shown live during `fit()`) negates
    `y_pred` internally (Cox log-hazard convention) and is therefore
    *anti-concordant* relative to the loss being optimized. Under
    correct fitting with a strong signal it drifts toward
    `1 - true_c_index` (i.e. *below* 0.5).
- **If the live training c-index appears below 0.5 on a survival
  model, your training is likely fine.** Trust the post-eval
  `patient_c_index` in `results_log.csv` for a faithful score.
- Resolving this internal inconsistency is tracked separately and is
  out of scope for 3.1.0.

#### Augmentation distributions (training reproducibility)

- **`random_jpeg_compression` quality formula** was wrong outside the
  default `q_min=50, q_max=100`. Default-config users see no
  difference; non-default bounds (e.g., the `q_min=30, q_max=100`
  default in `compose_color_distortion`) produced a wrong distribution
  and now produce the documented one.
- **`RandomGaussianBlur.calc_kernel`** hard-overrode `sigma=0.5`, so
  every blur kernel was 3×3 regardless of configured sigma. Users with
  non-default sigma will now see actual blur footprint scaling
  (sigma=2.0 → 9×9 kernel).
- **`StainNormalizer.augmented_transform` (Reinhard)** was broadcast-
  mismatched on `(3, 1)` target tensors after `fit()`; now squeezes to
  1D first.
- **Bezier color-distortion** — input value 255 now correctly maps to
  the last segment (was wrapping back to segment 0).

#### MIL training & inference

- **TransMIL attention overlays** were off-by-one. The per-tile slice
  forgot a CLS token at sequence position 0; every overlay was shifted
  by one tile and the last patch was dropped. Predictions were
  unaffected. Re-run `predict_mil` / `eval` to regenerate `*_att.npz`
  files. Trained models do not need to be retrained.
- **MIL `TrainerConfig.aggregation_level`** was lost on JSON save/load
  round-trip — `'patient'` setting silently reverted to `'slide'` on
  reload.
- **`predict_multimodal_mil(uq=True)`** previously crashed with
  `ValueError: too many values to unpack`. Now correctly emits per-
  outcome `uncertainty{i}` columns.
- **MIL multimodal regression mode** (no categorical classes) — the
  training summary log statement crashed on `unique=None`; now skips
  the categories line.
- **MIL training class-weight construction** no longer crashes when a
  class is absent from the training split (e.g., a stratified split
  that misses a rare class).
- **MIL encoder fix** — `Encoders require their input argument must be
  uniformly strings or numbers. Got ['list']` resolved by switching
  list-of-arrays handling in `mil/utils.py::get_labels`.
- **`MILFeatures` model dispatch** — `MIL_fc` and `MIL_fc_mc` were
  matched as a single substring (typo); now correctly dispatched.

#### Slide loader, QC, and visualization

- **Strided QC edge-tile bounds** — bottom/right-edge tiles were
  silently cropped to the inner overlap-trimmed region. Now keep
  their full extent.
- **vips multipage TIFF `level_count`** — corrected from
  `min(N-3, 1)` to `max(N-3, 1)`; non-OpenSlide multipage TIFFs now
  expose all pyramid levels instead of clamping to 1.
- **cucim `read_region` non-padded path** was unreachable
  (`TypeError`); now executes correctly.
- **cucim resize axis order** was passing `(width, height)` to
  `skimage.transform.resize` (which expects `(height, width)`); the
  non-cv2 path produced transposed regions.
- **Otsu QC** — ROI mask cast to bool before inversion (was
  bitwise-on-float); skips `bitwise_or` when the mask is all-zeros to
  prevent zeroing the thumbnail.
- **DeepFocus mixed precision** was applied after model construction
  (no-op); now applied before, as intended.
- **Heatmap extent for `stride_div > 1`** — `calculate_heatmap_extent`
  was missing the trailing offset on right/bottom; high-overlap
  heatmaps were visually compressed.
- **Renderer dropout reduction in Studio** — `yp_drop[0]` was returned
  for every outcome; now correctly returns `yp_drop[n]` per outcome.

#### Dataset / Project / Mosaic

- **`Dataset.find_slide(patient=...)` / `find_tfrecord(patient=...)`**
  was filtering on the wrong key; now correctly filters by patient.
- **`Dataset.check_duplicates`** MSE divisor was `H*W` (excluded the
  channel axis); now `A.size` (all elements). Numerical values change;
  relative ranking unchanged.
- **`Project.generate_heatmaps`** — `skip_completed` `return`→`continue`
  fix; completion-skip used to abort the whole batch on the first
  already-completed slide.
- **`Project.smac_search`** parameter state is now deep-copied per
  iteration (was leaking across iterations). Optimization results will
  differ.
- **SMAC `early_stop` parameter** — was never registered with the
  search space; users who configured `early_stop` saw default-encoded
  (no-early-stop) behavior.
- **`Mosaic` generation in `coords` and `tuples` modes** — was
  effectively broken (downstream code expected a DataFrame schema that
  was never built); now works correctly.
- **`_setup_input_labels`** (categorical input vector) — input width
  is now harmonized between train and validation sets. ⚠️ Models
  trained pre-3.1.0 with val-only categories may not load against
  post-3.1.0 inference unless the val set matched.
- **`_detect_classes_from_labels` for DataFrame integer labels** —
  changed from `unique.max() + 1` to `len(unique)`. Affects classes
  with non-zero-indexed or sparse integer labels.

#### Stats / metrics

- **AUROC / c-index logging** — 0.0 values were treated as "not
  computed"; now distinguishes None from 0.0.
- **Plot subsamples** — `np.random.choice(..., replace=False)` so
  histograms and scatters no longer skewed by duplicate indices.
- **`SlideMap.neighbors()`** uses `.iloc` (was `.loc`); post-`filter()`
  results were unreliable.
- **`SlideMap.save_range_clip` / `load_range_clip`** round-trip — was
  broken; now uses `os.path.join` + resolved load path.

#### Studio (imgui)

- ROI deep-copy semantics, color preservation across ROI merges,
  refresh-labels after add-hole, selected-ROI index bounds guard.
- Renderer dropout uncertainty per-outcome correct (see math fixes
  above).
- `_render_manager.set_async()` actually tears down workers on
  sync↔async toggle (was previously only flipping a flag).
- Prediction summaries no longer crash on fully-masked overlays
  (NaN-safe via `np.nanmean`).

### Fixed (silent / non-behavior-changing hardening)

A large rollup of correctness fixes across `Dataset`, `Project`,
`mosaic`, `heatmap`, `model/{tensorflow,torch,features,base}`,
`model/extractors/*`, `slide/{backends,qc,report,utils,wsi}`,
`io/{gaussian,io_utils,tensorflow,torch/*}`, `tfrecord/{torch/dataset,
writer}`, `gan/*`, `norm/*`, `stats/*`, `studio/*`, `util/*`. Highlights:

- **Resource-leak fixes** — `MultiTFRecordDataset` re-iter file
  handles, multiprocessing pools now `pool.close() + pool.join()`,
  `tfrecord2idx` file handles via `try/finally`, segment buffer
  try/finally.
- **Trainer / model-path correctness (TF)** — epoch resume numbering,
  early-stop final-evaluation reachability, slide-only model graph,
  frozen-layer index clamp, single-outcome classification load,
  feature-extractor channel detection under no-pooling.
- **PyTorch Trainer** — UnboundLocalError on explicit
  `validation_batch_size`; UncertaintyInterface UQ feature collection
  was broken for `len(layers) > 1` (TypeError on `range(self.layers)`).
- **`weights_only=True`** is enforced on all `torch.load` calls in core
  paths to close the pickle-RCE vector on untrusted checkpoints.
- **Studio `MILModelConfig.inspect_batch`** — multimodal+no-`use_lens`
  branch corrected (was iterating rows of bag 0 instead of iterating
  bags; production multimodal models all set `use_lens=True` so this
  path was unreachable in practice).
- **NumPy 2.x compatibility** — `np.fromstring` → `np.frombuffer`
  across slide_test, norm, _renderer, mosaic, _slide/_torch;
  `worker_init_fn` uses `int(...)` coercion to avoid uint32 overflow.
- **GAN interpolator** — tile-size mismatch warning condition was
  always-False (typo); features unpacking checked the wrong shape;
  noise tensors now use float32 (was float64).
- **Studio prediction summaries** no longer crash on fully-masked
  overlays.
- **Plugin loader** is now defensive (catches load/register exceptions
  and continues).
- **`UnrecognizedBackendError`** accepts custom message arguments.
- **Many small KeyError / AttributeError / UnboundLocalError guards**
  across the codebase.

(See `git log 3.0.2..3.1.0` for the complete per-file change list.)

---

## 3.0.2 — 2024-10-18

(See git tag `3.0.2`.)
