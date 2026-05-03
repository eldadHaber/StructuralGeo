# Expected-output notes

- `lithology_label.npy` mirrors `ground_truth/lithology_volume.npy`.
  The pipeline's lithology head should reproduce this within its
  acceptance criterion (label-recall on rock voxels).
- `stratigraphic_label.npy` is identical to lithology_label here
  (lithology = stratigraphy mapping in v1 per spec).
- `fault_distance.npy` is the per-voxel Euclidean distance to the
  nearest ground-truth fault plane (extruded down-dip). The pipeline
  output should agree within ~2 mesh cells (acceptance criterion 9).
