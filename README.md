# DeepStream Coding Agent

A project showcasing how to leverage **AI coding assistants** (Cursor, Claude Code, etc.) for accelerated [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) application development using a curated agentic skill and structured prompts.

> **Disclaimer:** Code generated with AI coding assistants is intended as a development starting point. All generated code must undergo your full software development lifecycle (SDLC) — including code review, testing, and security validation — before production use.

---

## Prerequisites

### For code generation (using the skill and prompts)

- **AI coding assistant** that supports agentic skills (e.g., Cursor, Claude Code)

No GPU, SDK, or special hardware is required — the skill and example prompts work on any system.

### For running the generated code

The following are required on the target execution environment:

- **NVIDIA DeepStream SDK 9.0** — installed locally or available via [NVIDIA NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream)
- **Python 3.12+** with the `pyservicemaker` package
- **NVIDIA GPU** with driver version 590 or later
- **CUDA 13.1** and **TensorRT 10.14.1.48**
- **Supported OS:** Ubuntu 24.04 (x86_64 or ARM64/Jetson)

> For detailed environment setup, refer to the [DeepStream SDK Developer Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/).

---

## Project Structure

```
DeepStream-Coding-Agent/
├── skills/                  # Agentic skills for guided DeepStream development
│   └── deepstream-dev/      # DeepStream development skill with condensed references
├── example_prompts/         # Pre-built prompts for code generation
├── LICENSE                  # CC-BY-4.0 AND Apache-2.0
└── README.md                # This file
```

---

## Purpose

This project provides the tooling and reference material needed to:
1. **Supply domain-specific context** to AI coding assistants through a curated agentic skill
2. **Generate production-ready DeepStream code** using well-structured prompts
3. **Accelerate development** of video analytics pipelines with AI assistance

---

## Agentic Skill

An **agentic skill** is a structured knowledge package that an AI coding assistant can automatically discover and activate during code generation. It contains domain-specific rules, reference documentation, and guardrails that guide the AI agent to produce accurate, idiomatic code — without the developer needing to manually reference files in every conversation.

The `skills/deepstream-dev/` directory contains a DeepStream agentic skill that follows the standard `SKILL.md` convention supported by AI coding assistants such as Cursor, Claude Code, and others.

### About the Skill

This skill targets NVIDIA DeepStream SDK 9.0 development using the Python `pyservicemaker` API. When activated, it instructs the AI agent to consult bundled reference documents before generating any code, significantly reducing inaccuracies and ensuring correct API usage.

**Bundled reference topics:**

| Reference | Coverage |
|-----------|----------|
| `gstreamer_plugins.md` | GStreamer plugin properties |
| `service_maker_api.md` | Pipeline/Flow API, metadata access, probes |
| `use_cases_pipelines.md` | Pipeline patterns: playback, multi-inference, cascaded GIE |
| `kafka_messaging.md` | Kafka/message broker setup and configuration |
| `best_practices.md` | Design patterns, pitfalls, anti-patterns |
| `buffer_apis.md` | BufferProvider/Feeder and BufferRetriever/Receiver |
| `media_extractor_advanced.md` | MediaExtractor, MediaChunk, FrameSampler |
| `utilities_config.md` | PerfMonitor, EngineFileMonitor, SourceConfig |
| `nvinfer_config.md` | nvinfer config file format and all parameters |
| `tracker_config.md` | nvtracker config (NvDCF, IOU, DeepSORT, NvSORT) |
| `troubleshooting.md` | Error messages and solutions |
| `rest_api_dynamic.md` | REST API, dynamic source management |
| `docker_containers.md` | Docker images, Dockerfile examples, pyservicemaker install, container run commands |

### Installing the Skill

Copy the `deepstream-dev` skill directory (including its `references/` subdirectory) into the skills folder recognized by your AI coding assistant. You can install it at the **user level** (available across all projects) or at the **workspace level** (scoped to a single project).

| Tool | User-level path | Workspace-level path |
|------|----------------|---------------------|
| Cursor | `~/.cursor/skills/deepstream-dev/` | `<workspace>/.cursor/skills/deepstream-dev/` |
| Claude Code | `~/.claude/skills/deepstream-dev/` | `<workspace>/.claude/skills/deepstream-dev/` |
| Other tools | Consult your tool's documentation for the skills directory location |

#### Step 1: Create the Skills Directory

```bash
# Example: Cursor user-level
mkdir -p ~/.cursor/skills/

# Example: Claude Code user-level
mkdir -p ~/.claude/skills/

# Example: workspace-level (replace .cursor with your tool's directory)
mkdir -p <workspace>/.cursor/skills/
```

#### Step 2: Copy the Skill

```bash
# User-level (replace path with your tool's skills directory)
cp -r skills/deepstream-dev ~/.cursor/skills/

# Or workspace-level
cp -r skills/deepstream-dev <workspace>/.cursor/skills/
```

After copying, the directory structure should look like:

```
<skills-directory>/
└── deepstream-dev/
    ├── SKILL.md              # Skill definition with rules and quick references
    └── references/           # Condensed reference documents
        ├── best_practices.md
        ├── buffer_apis.md
        ├── gstreamer_plugins.md
        ├── kafka_messaging.md
        ├── media_extractor_advanced.md
        ├── nvinfer_config.md
        ├── rest_api_dynamic.md
        ├── service_maker_api.md
        ├── tracker_config.md
        ├── troubleshooting.md
        ├── use_cases_pipelines.md
        ├── utilities_config.md
        └── docker_containers.md
```

#### Step 3: Verify the Installation

1. Open (or restart) your AI coding assistant.
2. Open the agent / chat panel.
3. Ask a DeepStream-related question, for example:
   ```
   Create a DeepStream pipeline that reads a video file and runs object detection using ResNet18.
   ```
4. The agent should automatically activate the `deepstream-dev` skill and consult its reference documents before generating code.

> **Tip:** The skill is most effective in **Agent mode**. In agent mode, the AI assistant automatically selects and activates relevant skills based on the task context — no manual file referencing needed.

---

## Using Example Prompts

The `example_prompts/` directory contains pre-built prompts for generating DeepStream applications. Each prompt file provides a complete specification that an AI agent can follow to produce working code.

> **Getting started?** Begin with `video_infer_app.md` for a minimal single-stream inference example, then progress to `multi_stream_tracker.md` for multi-stream and tracking capabilities.

### Available Prompts

| Prompt File | Purpose |
|-------------|---------|
| `multi_stream_tracker.md` | Multi-stream RTSP app with tracker and 2x2 tiled display |
| `rtvi_vlm_core_app.md` | Complete RTSP video processing app with VLM integration |
| `rtvi_vlm_openapi_spec.md` | FastAPI microservice with OpenAPI specification. Should be used after the core app is generated using @rtvi_vlm_core_app.md |
| `video_infer_app.md` | Basic video file inference with bounding box display |
| `video_object_count.md` | Video inference with object detection counting |
| `yolov26s_detection.md` | YOLOv26s model download, ONNX export, and custom parsing library |

---

### Step-by-Step Guide: Using Prompts

#### Step 1: Open the AI Chat / Agent Panel

Open the agent or chat panel in your AI coding assistant. Most tools provide a keyboard shortcut or sidebar icon for this.

#### Step 2: Reference the Prompt File

Use your tool's file-referencing feature (e.g., `@` mentions) to include the prompt file:

```
@example_prompts/rtvi_vlm_core_app.md
```

Or simply type `@` and start typing the filename to search.

#### Step 3: Execute the Prompt

**Option A: Direct execution**

Reference the file in the chat and instruct the agent to follow it:

```
Follow the instructions in @example_prompts/rtvi_vlm_core_app.md to generate the application.
```

**Option B: Incremental execution**

For complex prompts, break them into smaller steps:

```
Based on @example_prompts/rtvi_vlm_core_app.md, first implement the vLLM backend module.
```

Then follow up with:
```
Now implement the frame selection logic as described in the prompt.
```

#### Step 4: Review and Iterate

1. Review the generated code in the diff view.
2. Accept or reject individual changes.
3. Ask follow-up questions for refinements:
   ```
   Can you optimize the GPU memory usage in the generated stream_processor.py?
   ```

---

### Example Workflow: Generating the RTVI Application

Here's a complete workflow for generating the RTVI VLM application:

**1. Generate Core Application**
```
@example_prompts/rtvi_vlm_core_app.md

Generate the complete application following these instructions.
```

**2. Add FastAPI Microservice**
```
@example_prompts/rtvi_vlm_openapi_spec.md

Create the FastAPI server with all endpoints shown in @rtvi_vlm_openapi_spec.png
```

---

## Best Practices for AI-Assisted Development

### Writing Effective Prompts

1. **Be specific** — Include exact requirements, constraints, and expected outputs
2. **Reference context** — Use `@` mentions to include relevant files and documents
3. **Break down complex tasks** — Divide large features into smaller, focused prompts
4. **Include examples** — Show expected input/output formats when applicable
5. **Specify the deployment target** — Mention whether the application targets dGPU (x86_64) or Jetson (ARM64), as pipeline elements and sink choices may differ

### Iterating on Generated Code

1. **Review before accepting** — Always inspect generated pipelines for correct element linking and property values
2. **Test incrementally** — Run the pipeline after each major change rather than building the entire application at once
3. **Use the troubleshooting reference** — If a pipeline fails, ask the agent to consult `troubleshooting.md` for known error patterns
4. **Provide error output** — When debugging, paste the full GStreamer or DeepStream error log into the chat for more accurate fixes

---

## Demo Video

<a href="https://www.youtube.com/watch?v=ZQTX7MeN7mI" target="_blank" rel="noopener noreferrer">
  <picture>
    <img src="https://img.youtube.com/vi/ZQTX7MeN7mI/maxresdefault.jpg" alt="Build Vision AI Pipelines with DeepStream Coding Agents" width="560" style="border-radius:8px">
  </picture>
</a>

---

## Additional Resources

- [NVIDIA DeepStream SDK Developer Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [DeepStream Python Apps (GitHub)](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)
- [NVIDIA NGC DeepStream Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream)

---

## Contributing

This project is currently not accepting contributions.

---

## License

This project is licensed under [CC-BY-4.0 AND Apache-2.0](LICENSE).

SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
