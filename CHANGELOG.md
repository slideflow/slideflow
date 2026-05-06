# Changelog

All notable changes to Slideflow are documented in this file. See git
history for the full list of commits behind each entry.

## 3.0.3 — 2026-05-06

A focused bug-fix and stability release on top of 3.0.2. No new
features beyond a couple of minor additions noted below; no API
changes; no data-format changes. Larger refactor work continues on
the v4 development line.

### Added

- **H-optimus0 feature extractor** — new `hoptimus0` extractor
  registered for use with MIL training and inference.
- **Weights-hash validation for `virchow` and `hoptimus0`** — extractor
  weights are now hash-checked at load time to catch corrupt
  downloads.
- **`TileMaskDataset.split(train_size, seed)`** — convenience method
  that splits a segmentation dataset into train/val subsets, keyed by
  slide.
- **Studio: copy/paste ROIs** — keyboard-based copy/paste of ROIs
  between slides in Slideflow Studio.

### Changed

- **MPS device default for feature extractors** — when a Mac with
  Apple Silicon is detected, feature extractors default to MPS instead
  of CPU. Pass `device='cpu'` explicitly to opt out.
- **Segmentation training switched to AdamW** — the default optimizer
  for segmentation model training is now AdamW; loss/throughput
  characteristics may differ slightly from prior runs.

### Fixed

#### Studio (imgui)

- ROI bug fixes: deep-copy semantics, color preservation across ROI
  merges, and miscellaneous ROI-edit edge cases.
- Heatmap bug fixes for tile-based (non-MIL) models.
- MIL bundle loading is now tolerant of unknown future param keys
  (`mil_config(..., validate=False)` on the Studio load path) — v3
  Studio no longer crashes when shown a bundle written by a future
  Slideflow version that adds new `mil_params.json` keys.
- Defensive fixes in renderer dropout indexing and render-manager
  async teardown paths.

#### MIL training & inference

- **Multimodal model regression mode** — `build_multimodal_learner`
  no longer crashes on `unique=None` (regression targets); the
  training summary now skips the "Unique categories" line when not
  applicable.
- **Encoder uniform-input fix** — `MIL bug fix: Encoders require their
  input argument must be uniformly strings or numbers. Got ['list']`
  resolved by switching `all_unique.append(_unique)` to
  `all_unique += _unique` in `mil/utils.py::get_labels`.
- Three additional latent bugs in `mil/data.py`, `mil/features.py`,
  and `norm/torch/reinhard.py` surfaced during a coverage push.

#### Feature extractors

- `rebuild_extractor` robustness fix.
- ViT kwargs forwarding + `map_location` fix on weights load.
- Drop duplicate factory registrations.
- Many small correctness fixes in `extractors/_factory.py`,
  `extractors/_factory_torch.py`, `extractors/_registry.py`,
  `extractors/vit.py`, `extractors/_slide/_torch.py`,
  `extractors/_tensorflow_base.py` (see "Bug fix cycle 1" below).

#### Segmentation

- Multi-GPU training hang resolved by setting `sync_dist=False` on the
  PyTorch Lightning logging path.
- Improved segmentation training logging.

#### Dataset, Project, and supporting machinery

A large rollup of correctness fixes from "bug fix cycle 1"
(36 sub-batches), covering — non-exhaustively:

- `Dataset.find_slide` / `find_tfrecord`: patient lookup was filtering
  on the wrong key; now resolves correctly.
- `Dataset.build_index`: `pool.close()` no longer crashes with
  `NameError` when `num_workers=0`; pool size now respects the
  resolved worker count.
- `Dataset.num_tiles`: manifest is now refreshed before the post-update
  list comprehension, eliminating a `KeyError` on newly added
  tfrecords.
- `Dataset.remove_filter`: `filter_blank` removal now works (was a
  silent no-op due to an unreachable `elif`).
- `Dataset.verify_img_format`: switched to in-order `imap` so the
  per-file mismatch diagnostic is accurate.
- `Project.generate_rois`: per-slide `dest` reassignment no longer
  funnels later slides into the first source's ROI directory when
  `dest=None`.
- `Project.generate_heatmaps`: `skip_completed` now `continue`s instead
  of `return`ing, so a completed-slide skip no longer kills the rest
  of the run.
- `Project.smac_search`: `models_dir` mutation is now restored under
  exceptions via `try/finally`, preventing project-state corruption on
  optimizer failure.
- `Project._get_smac_runner`: `ModelParams.load_dict` reuse no longer
  leaks state across SMAC iterations.
- `get_tfrecord_locations`: removed an `elif` that unconditionally
  rebuilt indexes on every call (defeating caching and racing
  concurrent readers).
- `mosaic`: coordinate/tuple mode fixes, and CSV cleanups.
- `heatmap`: extent computation for `stride_div > 1`; correctness
  fixes throughout.
- `model/base`: `ModelParams` validation hardening.
- `model/features`: `KeyError` guards and plumbing fixes.
- `model/tensorflow`: `Features` initialization, `Trainer`
  epochs/labels, and `add_regularization` race fixed.
- `model/tensorflow_utils`: Cox loss sort + variance divisor fixes.
- `model/torch`: dataloader + UQ range fixes.
- `model/torch_utils`: `print_module_summary` shape typo.
- `stats/metrics` (five fixes), `stats/plot` (subsample-without-
  replacement).
- `util/neptune_utils`: drop double workspace prefix in `run_loc`.
- `util/smac_utils`: early-stop registration + empty-list guards.
- `util/tfrecord2idx`: close file handles on exception.
- `project_utils`: k-fold regex + file leak fixes.
- Removed unused `slideflow/model/adv_utils.py`.
- Switched from deprecated `imghdr` to `filetype`.
- Several `np.fromstring` deprecation fixes.

#### Compatibility

- NumPy compatibility fix in feature extraction.

### Removed

- `slideflow/model/adv_utils.py` — unused; deleted as part of the
  cycle 1 cleanup.

### Skipped (not in 3.0.3, deferred to a later release)

For transparency, the following items were considered but held for a
future release because they require more careful porting:

- `BufferedTileMaskDataset` and the rest of the segmentation buffer
  arc — feature in development, not yet stabilized.
- A second bug-fix rollup ("cycle 2") covering TransMIL attention
  alignment, MIL `aggregation_level` save/load, multimodal UQ path,
  random-augmentation distribution fixes, NumPy worker-seed collision,
  and other behavior-change-flagged fixes — held for separate review.
