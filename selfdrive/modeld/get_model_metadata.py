#!/usr/bin/env python3
"""
Model Metadata Extraction for E2E Models

This script extracts metadata from ONNX model files using tinygrad's ONNX parser.
The extracted metadata includes:
- input_shapes: Shape information for all model inputs
- output_shapes: Shape information for all model outputs  
- output_slices: Named slices for interpreting model outputs
- model_checkpoint: Checkpoint identifier

This metadata is used by planners to dynamically map model outputs
without hardcoded index constants, supporting the Unified Neural Execution architecture.

Usage:
    python3 selfdrive/modeld/get_model_metadata.py /path/to/model.onnx
    
Output:
    Creates a .pkl file alongside the model containing extracted metadata.
"""
import sys
import pathlib
import codecs
import pickle
from typing import Any, Optional

try:
    from tinygrad.nn.onnx import OnnxPBParser
except ImportError as e:
    print(f"Error: tinygrad is required for metadata extraction")
    print(f"Import error: {e}")
    sys.exit(1)


class MetadataOnnxPBParser(OnnxPBParser):
  """Extended ONNX parser that extracts model metadata properties."""
  
  def _parse_ModelProto(self) -> dict:
    obj: dict[str, Any] = {"graph": {"input": [], "output": []}, "metadata_props": []}
    for fid, wire_type in self._parse_message(self.reader.len):
      match fid:
        case 7:
          obj["graph"] = self._parse_GraphProto()
        case 14:
          obj["metadata_props"].append(self._parse_StringStringEntryProto())
        case _:
          self.reader.skip_field(wire_type)
    return obj


def get_name_and_shape(value_info: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
  """Extract name and shape from ONNX value info."""
  shape = tuple(int(dim) if isinstance(dim, int) else 0 for dim in value_info["parsed_type"].shape)
  name = value_info["name"]
  return name, shape


def get_metadata_value_by_name(model: dict[str, Any], name: str) -> Optional[str]:
  """Retrieve metadata property by name."""
  for prop in model["metadata_props"]:
    if prop["key"] == name:
      return prop["value"]
  return None


def validate_metadata(metadata: dict) -> bool:
  """Validate extracted metadata has required fields."""
  required_fields = ['input_shapes', 'output_shapes']
  for field in required_fields:
    if field not in metadata:
      return False
    if not isinstance(metadata[field], dict):
      return False
  return True


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python3 get_model_metadata.py <model.onnx>")
    print("Extracts metadata from ONNX model and saves to <model>_metadata.pkl")
    sys.exit(1)
    
  model_path = pathlib.Path(sys.argv[1])
  
  if not model_path.exists():
    print(f"Error: Model file not found: {model_path}")
    sys.exit(1)
  
  print(f"Extracting metadata from: {model_path}")
  
  try:
    model = MetadataOnnxPBParser(model_path).parse()
  except Exception as e:
    print(f"Error parsing model: {e}")
    sys.exit(1)
  
  # Extract output_slices - critical for E2E model output mapping
  output_slices_raw = get_metadata_value_by_name(model, 'output_slices')
  if output_slices_raw is None:
    print("Warning: output_slices not found in model metadata")
    print("Model may not be compatible with E2E longitudinal control")
    output_slices = {}
  else:
    try:
      output_slices = pickle.loads(codecs.decode(output_slices_raw.encode(), "base64"))
    except Exception as e:
      print(f"Warning: Failed to decode output_slices: {e}")
      output_slices = {}
  
  metadata = {
    'model_checkpoint': get_metadata_value_by_name(model, 'model_checkpoint'),
    'output_slices': output_slices,
    'input_shapes': dict(get_name_and_shape(x) for x in model["graph"]["input"]),
    'output_shapes': dict(get_name_and_shape(x) for x in model["graph"]["output"]),
    # Additional metadata for E2E support
    'tinygrad_version': 'compatible',  # Mark as tinygrad-extracted
  }
  
  if not validate_metadata(metadata):
    print("Warning: Extracted metadata validation failed")
    print(f"  input_shapes present: {'input_shapes' in metadata}")
    print(f"  output_shapes present: {'output_shapes' in metadata}")
  
  # Save metadata to pickle file
  metadata_path = model_path.parent / (model_path.stem + '_metadata.pkl')
  try:
    with open(metadata_path, 'wb') as f:
      pickle.dump(metadata, f)
    print(f'Successfully saved metadata to: {metadata_path}')
    print(f'  input_shapes: {len(metadata["input_shapes"])} inputs')
    print(f'  output_shapes: {len(metadata["output_shapes"])} outputs')
    print(f'  output_slices: {len(metadata["output_slices"])} slices')
  except Exception as e:
    print(f"Error saving metadata: {e}")
    sys.exit(1)
