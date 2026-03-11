# nvtracker Configuration Reference

## Overview

The `nvtracker` GStreamer plugin provides multi-object tracking capabilities in DeepStream pipelines. It tracks objects detected by inference engines across video frames, assigning unique tracking IDs and maintaining object trajectories.

## Prerequisites

### Required System Dependencies

The tracker library (`libnvds_nvmultiobjecttracker.so`) requires the **libmosquitto** library for MQTT-based communication features. This must be installed before using the tracker.

**Install on Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libmosquitto1
```

**Install on RHEL/CentOS:**
```bash
sudo yum install mosquitto
```

**Common Error if Missing:**
```
gstnvtracker: Failed to open low-level lib at /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
dlopen error: libmosquitto.so.1: cannot open shared object file: No such file or directory
gstnvtracker: Failed to initialize low level lib.
```

If you see this error, install libmosquitto1 as shown above.

---

## Core Components

### nvtracker Element

The main GStreamer element that handles tracking.

### Low-Level Tracker Libraries

DeepStream provides multiple tracker implementations:

| Library | File | Description | Use Case |
|---------|------|-------------|----------|
| NvDCF | `libnvds_nvmultiobjecttracker.so` | Deep Correlation Filter tracker | High accuracy, moderate speed |
| IOU | `libnvds_nvmultiobjecttracker.so` | Intersection-Over-Union tracker | Fast, simple scenes |
| DeepSORT | `libnvds_nvmultiobjecttracker.so` | Deep learning-based tracker | Re-identification support |
| NvSORT | `libnvds_nvmultiobjecttracker.so` | NVIDIA optimized SORT | Balanced performance |

**Library Location**: `/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so`

---

## GObject Properties

### Required Properties

| Property | Type | Description |
|----------|------|-------------|
| `ll-lib-file` | string | Path to low-level tracker library |
| `ll-config-file` | string | Path to tracker configuration file |

### Optional Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `tracker-width` | int | 0 | Tracker input width (0=auto from inference) |
| `tracker-height` | int | 0 | Tracker input height (0=auto from inference) |
| `gpu-id` | int | 0 | GPU device ID |
| `display-tracking-id` | int | 1 | Show tracking ID in OSD (0/1) |
| `tracking-id-reset-mode` | int | 0 | ID reset on stream reset/EOS (0-3) |
| `tracking-surface-type` | int | 0 | Surface type for tracking |
| `input-tensor-meta` | int | 0 | Use tensor metadata from upstream |
| `tensor-meta-gie-id` | int | -1 | GIE ID for tensor metadata |
| `user-meta-pool-size` | int | 16 | Tracker user metadata buffer pool |
| `sub-batches` | string | - | Sub-batch configuration |
| `sub-batch-err-recovery-trial-cnt` | int | 3 | Max reinit trials on error |

### Usage Example

```python
pipeline.add("nvtracker", "tracker", {
    "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    "ll-config-file": "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
    "tracker-width": 640,
    "tracker-height": 384,
    "gpu-id": 0,
    "display-tracking-id": 1
})
```

---

## Tracker Configuration File

The tracker configuration file (YAML format) defines the tracking algorithm behavior.

### Basic Configuration Structure

```yaml
%YAML:1.0
NvMultiObjectTracker:
  # Global settings
  useUniqueID: 1
  maxTargetsPerStream: 150
  enableBatchProcess: 1
  
  # Tracker algorithm selection
  NvDCF:
    # NvDCF-specific settings
    
  # OR
  IOU:
    # IOU-specific settings
    
  # OR  
  DeepSORT:
    # DeepSORT-specific settings
```

---

## Tracker Algorithm Configurations

### NvDCF (Deep Correlation Filter)

Best for: High accuracy tracking with moderate computational cost.

```yaml
%YAML:1.0
NvMultiObjectTracker:
  useUniqueID: 1
  maxTargetsPerStream: 150
  enableBatchProcess: 1
  
  NvDCF:
    # Filter settings
    filterLr: 0.15
    filterChannelWeightsLr: 0.22
    filterChannelWeightsStr: 0.75
    featureChannelWeightsLr: 0.22
    featureChannelWeightsStr: 0.75
    
    # Search and target settings
    searchRegionPaddingScale: 3.0
    filterLrScale: 0.1
    targetSearchLrScale: 0.1
    
    # Visual tracking
    useColorNames: 1
    useHog: 1
    
    # Max consecutive frames to track without detection
    maxShadowTrackingAge: 30
    
    # Track confirmation
    probationAge: 3
    earlyTerminationAge: 1
    
    # Feature extraction
    featureImgSizeLevel: 1
    maxBatchSize: 30
    
    # State estimation
    stateEstimatorType: 1  # 0=Simple, 1=Kalman
    
    # Data association
    dataAssociatorType: 1  # 0=Simple, 1=Mahalanobis
    
    # Visual feature similarity
    visualSimilarityWeight: 0.3
```

**Sample Config Location**: `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml`

### IOU Tracker

Best for: Fast tracking in simple scenes, real-time requirements.

```yaml
%YAML:1.0
NvMultiObjectTracker:
  useUniqueID: 1
  maxTargetsPerStream: 150
  enableBatchProcess: 1
  
  IOU:
    # IOU threshold for matching
    iouThreshold: 0.1
    
    # Track management
    maxShadowTrackingAge: 30
    probationAge: 3
    earlyTerminationAge: 1
    
    # State estimation
    stateEstimatorType: 1  # 0=Simple, 1=Kalman
    
    # Data association
    dataAssociatorType: 0  # 0=Simple IOU
```

### DeepSORT

Best for: Re-identification across camera views, occlusion handling.

```yaml
%YAML:1.0
NvMultiObjectTracker:
  useUniqueID: 1
  maxTargetsPerStream: 150
  enableBatchProcess: 1
  
  DeepSORT:
    # Re-ID model configuration
    reidModelPath: /path/to/reid_model.onnx
    reidModelInputWidth: 128
    reidModelInputHeight: 256
    reidModelBatchSize: 100
    reidModelColorFormat: 0  # 0=RGB
    
    # Matching thresholds
    minMatchingScore: 0.3
    maxCosineDist: 0.3
    
    # Track management
    maxShadowTrackingAge: 30
    probationAge: 3
    earlyTerminationAge: 1
    
    # State estimation
    stateEstimatorType: 1  # 0=Simple, 1=Kalman
    
    # Data association
    dataAssociatorType: 2  # 2=Deep association
```

### NvSORT

Best for: Balanced performance, NVIDIA-optimized SORT.

```yaml
%YAML:1.0
NvMultiObjectTracker:
  useUniqueID: 1
  maxTargetsPerStream: 150
  enableBatchProcess: 1
  
  NvSORT:
    # IOU matching
    iouThreshold: 0.1
    
    # Track management
    maxShadowTrackingAge: 30
    probationAge: 3
    earlyTerminationAge: 1
    
    # State estimation
    stateEstimatorType: 1  # Kalman filter
    
    # Data association
    dataAssociatorType: 0
```

---

## Complete Configuration Examples

### Example 1: High-Performance NvDCF

```yaml
%YAML:1.0
NvMultiObjectTracker:
  # Global settings
  useUniqueID: 1
  maxTargetsPerStream: 150
  enableBatchProcess: 1
  
  # Enable past frame data for analytics
  enablePastFrame: 1
  pastFramePoolSize: 300
  
  # Terminate track on stream reset
  terminateTrackerOnStreamReset: 1
  
  NvDCF:
    # Core filter parameters
    filterLr: 0.15
    filterChannelWeightsLr: 0.22
    filterChannelWeightsStr: 0.75
    featureChannelWeightsLr: 0.22
    featureChannelWeightsStr: 0.75
    
    # Feature settings
    useColorNames: 1
    useHog: 1
    featureImgSizeLevel: 1
    
    # Search region
    searchRegionPaddingScale: 3.0
    filterLrScale: 0.1
    targetSearchLrScale: 0.1
    
    # Track lifecycle
    maxShadowTrackingAge: 30
    probationAge: 3
    earlyTerminationAge: 1
    
    # Batching
    maxBatchSize: 30
    
    # State estimation: Kalman filter
    stateEstimatorType: 1
    
    # Data association: Mahalanobis
    dataAssociatorType: 1
    
    StateEstimator:
      processNoiseVar4Loc: 4.0
      processNoiseVar4Vel: 1.0
      measurementNoiseVar4Loc: 1.0
      measurementNoiseVar4Vel: 1.0
    
    DataAssociator:
      dataAssociationCostType: 0
      minDistThreshold4MatchedAssoc: 0.5
      minDistThreshold4UnmatchedAssoc: 0.25
      matchingScoreWeight4VisualSimilarity: 0.3
```

### Example 2: Fast IOU Tracker

```yaml
%YAML:1.0
NvMultiObjectTracker:
  useUniqueID: 1
  maxTargetsPerStream: 150
  enableBatchProcess: 1
  
  IOU:
    iouThreshold: 0.1
    maxShadowTrackingAge: 15
    probationAge: 2
    earlyTerminationAge: 1
    
    # Use Kalman filter for state estimation
    stateEstimatorType: 1
    
    # Simple IOU-based association
    dataAssociatorType: 0
    
    StateEstimator:
      processNoiseVar4Loc: 4.0
      processNoiseVar4Vel: 1.0
      measurementNoiseVar4Loc: 1.0
      measurementNoiseVar4Vel: 1.0
    
    DataAssociator:
      dataAssociationCostType: 0
      minDistThreshold4MatchedAssoc: 0.5
```

---

## Pipeline Integration

### Basic Usage

```python
from pyservicemaker import Pipeline
import platform

def tracking_pipeline(video_path, infer_config):
    pipeline = Pipeline("tracking-pipeline")
    
    # Source and decoding
    pipeline.add("filesrc", "src", {"location": video_path})
    pipeline.add("h264parse", "parser")
    pipeline.add("nvv4l2decoder", "decoder")
    pipeline.add("nvstreammux", "mux", {"batch-size": 1, "width": 1920, "height": 1080})
    
    # Inference
    pipeline.add("nvinfer", "pgie", {"config-file-path": infer_config})
    
    # Tracker
    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file": "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
        "tracker-width": 640,
        "tracker-height": 384
    })
    
    # Display
    pipeline.add("nvosdbin", "osd")
    sink_type = "nv3dsink" if platform.processor() == "aarch64" else "nveglglessink"
    pipeline.add(sink_type, "sink")
    
    # Link
    pipeline.link("src", "parser", "decoder")
    pipeline.link(("decoder", "mux"), ("", "sink_%u"))
    pipeline.link("mux", "pgie", "tracker", "osd", "sink")
    
    pipeline.start().wait()
```

### Accessing Tracking Data

```python
from pyservicemaker import BatchMetadataOperator

class TrackingAnalyzer(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            print(f"Frame {frame_meta.frame_number}:")
            
            for obj_meta in frame_meta.object_items:
                print(f"  Object: class={obj_meta.class_id}, "
                      f"object_id={obj_meta.object_id}, "
                      f"confidence={obj_meta.confidence:.2f}")
```

---

## Performance Tuning

### Tracker Dimensions

Match tracker dimensions to inference input for best performance:

```python
# Inference config uses 960x544
# Match tracker dimensions
pipeline.add("nvtracker", "tracker", {
    "ll-lib-file": "...",
    "ll-config-file": "...",
    "tracker-width": 960,
    "tracker-height": 544
})
```

### Batch Processing

Enable batch processing in config for multi-stream:

```yaml
NvMultiObjectTracker:
  enableBatchProcess: 1
  maxTargetsPerStream: 150
```

### Track Lifecycle Parameters

Adjust based on scene complexity:

| Scene Type | maxShadowTrackingAge | probationAge | earlyTerminationAge |
|------------|---------------------|--------------|---------------------|
| Simple | 15 | 2 | 1 |
| Moderate | 30 | 3 | 1 |
| Complex/Occlusion | 60 | 5 | 2 |

---

## Sample Configuration Files

DeepStream provides sample tracker configurations:

```
/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/
├── config_tracker_NvDCF_accuracy.yml    # High accuracy, slower
├── config_tracker_NvDCF_perf.yml        # Balanced performance
├── config_tracker_IOU.yml               # Fast IOU tracker
├── config_tracker_NvSORT.yml            # NVIDIA SORT
└── config_tracker_DeepSORT.yml          # Deep SORT with Re-ID
```

---

## Common Issues

### Issue 1: Tracking IDs Not Appearing

**Cause**: OSD not configured to display tracking IDs.

**Solution**: Set `display-tracking-id` property:
```python
pipeline.add("nvtracker", "tracker", {
    "display-tracking-id": 1,
    # ... other properties
})
```

### Issue 2: Frequent ID Switches

**Cause**: Low IOU threshold or short shadow tracking age.

**Solution**: Increase in config file:
```yaml
NvDCF:
  maxShadowTrackingAge: 45  # Increase from default
  iouThreshold: 0.3         # Higher threshold
```

### Issue 3: Too Many Simultaneous Tracks

**Cause**: Low confidence threshold in detector.

**Solution**: Increase detector threshold and/or add track filtering:
```yaml
NvMultiObjectTracker:
  maxTargetsPerStream: 50  # Reduce from default 150
```

### Issue 4: Tracking Performance Issues

**Cause**: Mismatched tracker dimensions.

**Solution**: Match tracker dimensions to inference:
```python
# If inference uses 960x544
pipeline.add("nvtracker", "tracker", {
    "tracker-width": 960,
    "tracker-height": 544,
    # ... other properties
})
```

---

## Related Documentation

- **GStreamer Plugins Overview**: `02_gstreamer_plugins_overview.md`
- **Service Maker Python API**: `03_service_maker_python_api.md`
- **Multi-Inference Use Case**: `05_use_case_multi_inference.md`
- **nvinfer Configuration**: `14_nvinfer_configuration_reference.md`
