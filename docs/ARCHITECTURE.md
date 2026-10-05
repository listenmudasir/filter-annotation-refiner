# Architecture

## Product invariant

Source YOLO annotations are read-only. The converter writes a separate segmentation dataset.

## Core path

`scan_dataset` → `Sam3Backend.candidates` → `SmartRefinementEngine.refine_annotation` → polygon extraction → atomic TXT write → reports/resume state.

## Geometry invariant

`refiner.geometry` treats a mask as a list of `MaskPolygon` parts, each with its own
holes — that is the lossless form. Flattening to a single ring is only ever an
*export* decision, taken in `parts_to_export_rings` and verified by rasterizing the
result and scoring it against the source mask. No stage may silently drop a component
or fill a hole; `polygon_fidelity` is the number that proves it, and it is persisted
per object so a regression shows up in the QA reports rather than in a trained model
months later.

All geometry work happens on a crop around the object, so contour extraction and the
simplification search scale with object size rather than image size. This is what
makes the stage usable on 4K frames.

## Why image-level caching matters

SAM image features are computed once in `set_image`. Every object in that image then reuses the same inference state for several box/point prompt candidates. This avoids re-running the vision encoder per object.

## Failure behavior

An image is written only after every annotation in that image has produced a valid polygon. If an object fails, the image is recorded in `reports/failures.csv` rather than emitting a silently incomplete label file.

## Review policy

The quality score triages labels; it does not certify correctness. Low-quality objects enter `review_queue.csv`, while high-quality labels are immediately available in the output dataset.
