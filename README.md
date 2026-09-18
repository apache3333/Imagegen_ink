<div align="center">


# AI Image Generator for Inkscape

[![Inkscape](https://img.shields.io/badge/Inkscape-1.0+-blue.svg)](https://inkscape.org/)
[![Python](https://img.shields.io/badge/Python-3.6+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Generate and edit images using AI directly within Inkscape**

A powerful Inkscape extension that integrates multiple AI image generation providers (OpenAI DALL-E, Stability AI, Replicate, Venice AI, and local models) to create, edit, and transform images without leaving your design workflow.

</div>


## 📺 Demo

<div align="center">

<!-- Replace VIDEO_ID with your actual YouTube video ID -->
[![Watch the Demo](https://img.youtube.com/vi/bw5SQ3BXnKg/maxresdefault.jpg)](https://www.youtube.com/watch?v=bw5SQ3BXnKg)

*Click to watch the full tutorial on YouTube*

</div>


---

## 📋 Table of Contents

- Features
- Requirements
- Installation
- Quick Start
- Usage Guide
- Configuration
- Provider Comparison
- Examples
- Troubleshooting
- Contributing
- License

---

## ✨ Features

- **🤖 Multiple AI Providers**
  | Provider | Best For | Features |
  |----------|----------|----------|
  | **OpenAI DALL-E** | General use, easy setup | DALL-E 2, DALL-E 3, gpt-image-1 |
  | **Stability AI** | High quality, control | SDXL, negative prompts, seeds |
  | **Replicate** | Latest models | Flux, SDXL, community models |
  | **Venice AI** ⚠️ *experimental* | Privacy, model choice | SD3.5, Qwen, Flux 2, Nano Banana, Seedream |
  | **Local** | Privacy, no API costs | Automatic1111, ComfyUI |

- **🎨 Four Operation Modes**
  | Mode | Description |
  |------|-------------|
  | **Generate** | Create new images from text prompts |
  | **Edit** | Modify parts of existing images with masks |
  | **Variation** | Create variations of selected images |
  | **Img2Img** | Transform images with prompts |

- **⚙️ Advanced Options**
  - Negative prompts (what to avoid)
  - Seed control for reproducibility
  - Batch generation (1-4 images)
  - Custom dimensions
  - CFG scale and sampling steps
  - Mask modes for targeted editing

- **📁 Smart File Management**
  - Auto-save to disk with timestamps
  - Embed images in SVG or link externally
  - Generation history tracking
  - Configurable save directories

- **🎯 Flexible Placement**
  - 8 position presets (center, corners, edges)
  - Selection-based positioning
  - Multiple scale modes (original, fit, custom)

---

## 📦 Requirements

### Core Requirements

| Component | Version | Purpose |
|-----------|---------|---------|
| **Inkscape** | 1.0+ | Vector graphics editor |
| **Python** | 3.6+ | Extension runtime |
| **certifi** | Latest | SSL certificate handling |

### Optional Dependencies

| Component | Purpose | Installation |
|-----------|---------|--------------|
| **Pillow** | Image editing/masking | `pip install Pillow` |

### API Keys (by provider)

| Provider | Get API Key |
|----------|-------------|
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Stability AI | [platform.stability.ai](https://platform.stability.ai/) |
| Replicate | [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens) |
| Venice AI | [venice.ai/settings/api](https://venice.ai/settings/api) |
| Local | No key needed |

---

## 🚀 Installation

### Step 1: Locate Your Inkscape Extensions Directory

**Windows:**
```
C:\Users\[YourUsername]\AppData\Roaming\inkscape\extensions\
```

**macOS:**
```
~/Library/Application Support/org.inkscape.Inkscape/config/inkscape/extensions/
```

**Linux:**
```
~/.config/inkscape/extensions/
```

> 💡 **Tip:** In Inkscape: **Edit → Preferences → System** shows the extensions path

### Step 2: Install the Extension

1. **Create the extension folder:**
   ```bash
   mkdir -p [extensions-directory]/image_maker
   ```

2. **Copy files to the folder:**
   ```bash
   cp img_gen.py [extensions-directory]/image_maker/
   cp img_gen.inx [extensions-directory]/image_maker/
   ```

3. **Install Python dependencies:**
   ```bash
   pip install certifi Pillow
   ```

4. **Restart Inkscape**

### Step 3: Verify Installation

Open Inkscape and check: **Extensions → Generate → AI Image Generator**

---

## 🚀 Quick Start

### Generate Your First Image

1. Open Inkscape
2. Go to **Extensions → Generate → AI Image Generator**
3. Select **Provider**: OpenAI
4. Enter your **API Key** (or configure in config file)
5. Set **Operation Mode**: Generate
6. Enter a **Prompt**: `A serene mountain landscape at sunset, digital art`
7. Click **Apply**

**Result:** AI-generated image appears on your canvas!

### Edit an Existing Image

1. Select an image in Inkscape
2. Open the extension
3. Set **Operation Mode**: Edit
4. Choose **Mask Mode**: Center (to edit center area)
5. Enter **Edit Instruction**: `Add a castle on the hill`
6. Click **Apply**

---

## 📖 Usage Guide

### Tab Overview

| Tab | Purpose |
|-----|---------|
| **Mode** | Select operation (generate, edit, variation, img2img) |
| **Provider** | Choose AI provider and configure API key |
| **Prompt** | Enter text description and negative prompts |
| **Settings** | Model, size, quality, style options |
| **Advanced** | Seed, batch count, CFG scale, steps |
| **Edit** | Mask modes and feathering for editing |
| **Save** | Output directory and embedding options |
| **Position** | Placement and scaling settings |

### Operation Modes

#### Generate Mode

Create new images from text descriptions.

**Settings:**
- **Prompt**: Describe what you want (be specific!)
- **Negative Prompt**: What to avoid
- **Model**: Choose from provider's available models
- **Size**: Image dimensions
- **Quality**: standard or hd (DALL-E 3)
- **Style**: vivid or natural (DALL-E 3)

**Example Prompt:**
```
A professional photograph of a modern office interior, 
natural lighting, minimalist design, plants, large windows
```

#### Edit Mode

Modify parts of an existing image using masks.

**Requirements:**
- Select an image first
- Provide edit instructions

**Mask Modes:**
| Mode | Area Edited |
|------|-------------|
| `full` | Entire image |
| `center` | Center 50% |
| `edges` | Outer edges only |
| `top_half` | Top portion |
| `bottom_half` | Bottom portion |
| `left_half` | Left portion |
| `right_half` | Right portion |

**Or:** Select shapes to use as custom masks!

> **Venice AI:** masks are applied locally rather than by the API — see
> [Venice AI Setup](#venice-ai-setup-experimental).

#### Variation Mode

Create variations of an existing image.

1. Select an image
2. Choose Variation mode
3. Click Apply
4. New similar image appears

> **Venice AI** does not support this mode — use Img2Img instead.

#### Img2Img Mode

Transform an image based on a prompt while maintaining structure.

**Key Setting:** `img2img_strength` (0-1) — *ignored by Venice AI*
- **0.3**: Subtle changes, keeps most details
- **0.5**: Moderate transformation
- **0.75**: Major changes, loose structure
- **1.0**: Complete regeneration

---

## ⚙️ Configuration

### API Key Configuration

Three ways to provide API keys (in priority order):

#### 1. Direct Input (Temporary)
Enter in the API Key field each time.

#### 2. Environment Variable (Recommended for security)

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
```

**Linux/macOS:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**Environment Variables by Provider:**
| Provider | Variable |
|----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Stability AI | `STABILITY_API_KEY` |
| Replicate | `REPLICATE_API_TOKEN` |
| Venice AI | `VENICE_API_KEY` |

#### 3. Config File (Persistent)

Edit `.ai_image_config.json` in the extension directory:

```json
{
    "openai_api_key": "sk-your-openai-key",
    "stability_api_key": "sk-your-stability-key",
    "replicate_api_key": "r8_your-replicate-token",
    "venice_api_key": "your-venice-api-key",
    "default_provider": "openai",
    "default_model": "dall-e-3",
    "default_size": "1024x1024",
    "default_quality": "standard",
    "default_save_directory": "C:/Users/YourName/Pictures/AI_Images"
}
```

### Proxy Configuration

For corporate networks or VPNs:

1. Check **Use Proxy**
2. Enter **Proxy URL**: `http://proxy.company.com:8080`

### Local Model Setup

For Automatic1111 or ComfyUI:

1. Start your local server (default: `http://127.0.0.1:7860`)
2. Select **Provider**: Local
3. Optionally set **Custom Endpoint** if using non-default port

### Venice AI Setup (Experimental)

1. Get a key at [venice.ai/settings/api](https://venice.ai/settings/api)
2. Select **Provider**: Venice AI
3. Provide the key in the API Key field, via `VENICE_API_KEY`, or as
   `venice_api_key` in the config file
4. On the **Model** tab pick a model marked `(Venice)`

> ⚠️ **Experimental.** Venice's own documentation describes its image edit endpoints as
> experimental and subject to change. The **Custom endpoint** field does not apply to
> Venice — it is Local-only.

**Supported modes**

| Mode | Venice | Notes |
|------|--------|-------|
| Generate | ✅ | `POST /image/generate` — honours negative prompt, seed, CFG scale and steps |
| Edit | ✅ | `POST /image/edit` — prompt-based, see masking below |
| Img2Img | ✅ | `POST /image/edit` — **Transformation strength is ignored** (no API equivalent) |
| Variation | ❌ | Venice has no variation endpoint; use Img2Img with a prompt instead |

**Models**

| Model | Sizing | Notes |
|-------|--------|-------|
| `venice-sd35` | width/height | Default. SD 3.5, up to 30 steps |
| `z-image-turbo` | width/height | Fastest, fixed at 8 steps |
| `qwen-image-3` | aspect ratio | Strong general purpose model |
| `flux-2-pro` | aspect ratio | Flux 2 |
| `nano-banana-2` | aspect ratio | Long prompts (32k characters) |
| `seedream-v5-lite` | aspect ratio | Inexpensive |
| `firered-image-edit` | — | Default for Edit and Img2Img |
| `qwen-image-3-edit` | — | Edit and Img2Img |
| `nano-banana-2-edit` | — | Edit and Img2Img, long prompts |

Selecting a non-Venice model while Venice is the provider falls back to a Venice default
and says so, rather than failing.

**Masking works differently on Venice**

Venice has **no mask channel** — its API rewrites the whole frame, and its documentation is
explicit that additional images act as whole-image references rather than region maps. So
this extension applies masks **locally**:

- **Mask region = "Full image"** → the Venice result is used as-is.
- **Any other mask mode, or "Use selected shapes as mask"** → Venice regenerates the whole
  frame, and the masked region is composited back over the original here, honouring
  **Feather radius** for the seam. Everything outside the mask is pixel-identical to the
  original; inside it is a fresh generation, which may not line up perfectly at the edges.
  This path needs **Pillow**, and says so if it is missing.

This is local compositing, not server-side inpainting.

**Sizing is mapped per model**

Venice models fall into two families, so the Size dropdown is translated rather than passed
through, and the output size may differ from what was requested:

- **Pixel-based models** (`venice-sd35`, `z-image-turbo`) take width and height, capped at
  **1280px** per side and rounded to the model's step. `1792x1024` becomes `1280x736`.
- **Aspect-ratio models** (everything else) reject explicit dimensions, so the closest
  supported aspect ratio is sent instead. The placement box is then matched to the image
  that comes back, so it is not stretched.

Sampling steps and seeds are clamped to each model's accepted range, with a note when that
happens. Negative prompts use Venice's real `negative_prompt` field when generating; the
edit endpoint has no such field, so a negative prompt is appended to the instruction as
`. Avoid: ...` instead. Venice's own defaults for watermarking and safe mode are left
untouched.

---

## 🔄 Provider Comparison

| Feature | OpenAI | Stability AI | Replicate | Venice AI | Local |
|---------|--------|--------------|-----------|-----------|-------|
| **Setup Difficulty** | Easy | Easy | Medium | Easy | Hard |
| **Cost** | ~$0.04/image | ~$0.01/image | Varies | ~$0.01-0.10/image | Free |
| **Negative Prompts** | Limited | ✅ Full | ✅ Full | ✅ Full (generate) | ✅ Full |
| **Seed Control** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Custom Sizes** | Limited | ✅ | ✅ | Mapped per model | ✅ |
| **Edit/Inpaint** | ✅ | ✅ | ❌ | Prompt-based, no mask channel | ✅ |
| **Variations** | ✅ | Via img2img | Via img2img | ❌ Use img2img | Via img2img |
| **Best Models** | DALL-E 3 | SDXL | Flux Pro | Qwen Image 3, Nano Banana 2 | Any |
| **Privacy** | Cloud | Cloud | Cloud | Cloud (not retained) | Local |

### Recommended Use Cases

- **OpenAI**: Quick prototypes, general imagery, beginners
- **Stability AI**: Detailed control, specific styles, production work
- **Replicate**: Access to Flux, experimental models
- **Venice AI**: Wide model choice behind one key, privacy-conscious cloud use
- **Local**: Privacy-sensitive, high volume, custom models

---

## 🎨 Presets

Quick-start with optimized settings:

| Preset | Style | Quality | Negative Prompt |
|--------|-------|---------|-----------------|
| `photorealistic` | natural | hd | cartoon, illustration, painting |
| `artistic` | vivid | hd | photo, realistic |
| `quick_draft` | natural | standard | (none) |
| `high_quality` | vivid | hd | low quality, blurry, distorted |

---

## 💡 Examples

### Example 1: Product Mockup

```
Provider: OpenAI (DALL-E 3)
Mode: Generate
Prompt: Professional product photography of a sleek smartphone 
        on a marble surface, soft studio lighting, minimalist 
        background, 4K quality
Size: 1024x1024
Quality: hd
Style: natural
```

### Example 2: Artistic Illustration

```
Provider: Stability AI (SDXL)
Mode: Generate
Prompt: Fantasy castle on floating island, dramatic sunset, 
        volumetric clouds, digital painting style
Negative: blurry, low quality, watermark, text
Size: 1216x832
CFG Scale: 8
Steps: 40
Seed: 42
```

### Example 3: Edit Background

```
Provider: OpenAI
Mode: Edit
Mask Mode: edges
Selected Image: Portrait photo
Edit Instruction: Replace background with tropical beach sunset
```

### Example 4: Style Transfer

```
Provider: Local (Automatic1111)
Mode: Img2Img
Prompt: oil painting, impressionist style, vibrant colors
Strength: 0.65
Selected Image: Photograph of city street
```

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><b>Extension not appearing in menu</b></summary>

**Solutions:**
1. Verify files are in correct location:
   ```
   [extensions]/image_maker/img_gen.py
   [extensions]/image_maker/img_gen.inx
   ```
2. Check file permissions (Linux/macOS):
   ```bash
   chmod +x img_gen.py
   ```
3. Restart Inkscape completely
4. Check error log: **Edit → Preferences → System → Open Error Log**

</details>

<details>
<summary><b>API key not found</b></summary>

**Error:** `No API key found for OpenAI DALL-E`

**Solutions:**
1. Enter key directly in the API Key field
2. Check environment variable is set:
   ```bash
   echo $OPENAI_API_KEY
   ```
3. Verify config file exists and has correct key:
   ```
   [extension-dir]/.ai_image_config.json
   ```
4. Ensure "Use config key" is checked

</details>

<details>
<summary><b>SSL certificate error</b></summary>

**Error:** `SSL: CERTIFICATE_VERIFY_FAILED`

**Solutions:**
1. Install/update certifi:
   ```bash
   pip install --upgrade certifi
   ```
2. **macOS**: Run Python certificate installer:
   ```bash
   /Applications/Python\ 3.x/Install\ Certificates.command
   ```
3. Use proxy settings if behind corporate firewall

</details>

<details>
<summary><b>Pillow not found (for editing)</b></summary>

**Error:** `PIL/Pillow library required for image editing`

**Solution:**
```bash
pip install Pillow
```

Or for Inkscape's bundled Python:
```bash
# Windows
"C:\Program Files\Inkscape\bin\python.exe" -m pip install Pillow

# macOS
/Applications/Inkscape.app/Contents/MacOS/python -m pip install Pillow
```

</details>

<details>
<summary><b>Connection timeout</b></summary>

**Solutions:**
1. Check internet connection
2. Verify API endpoint is accessible
3. For local: ensure server is running
4. Try increasing timeout in code (default: 180s)
5. Check if proxy is needed

</details>

<details>
<summary><b>Rate limit exceeded</b></summary>

**Error:** `429 Too Many Requests`

**Solutions:**
1. Wait and retry (extension has auto-retry with backoff)
2. Reduce batch_count
3. Upgrade API plan for higher limits
4. Use local provider for high volume

</details>

### Debug Mode

Enable detailed logging by checking generation history:

```
[extension-dir]/.ai_image_history.json
```

This shows recent operations with timestamps, prompts, and settings.

---

## 📁 File Structure

```
image_maker/
├── img_gen.py              # Main extension code
├── img_gen.inx             # Inkscape extension definition
├── README.md               # This file
├── .ai_image_config.json   # Configuration (auto-created)
└── .ai_image_history.json  # Generation history (auto-created)
```

---

## 🔧 Customization

### Adding Custom Presets

Edit the `PRESETS` dictionary in img_gen.py:

```python
PRESETS = {
    # ...existing presets...
    'my_custom_preset': {
        'style': 'vivid',
        'quality': 'hd',
        'negative_prompt': 'your default negative prompt'
    },
}
```

### Adding New Providers

Extend the `PROVIDERS` dictionary:

```python
PROVIDERS = {
    # ...existing providers...
    'my_provider': {
        'name': 'My Custom Provider',
        'generate_url': 'https://api.example.com/generate',
        'env_key': 'MY_PROVIDER_API_KEY',
        'config_key': 'my_provider_api_key',
        'models': ['model-1', 'model-2'],
        'sizes': ['1024x1024', '512x512']
    },
}
```

Then implement the corresponding `generate_my_provider()` method.

### Changing Default Settings

Edit the config file or modify `default_config` in `load_config()`:

```python
default_config = {
    'default_provider': 'stability',  # Change default provider
    'default_model': 'stable-diffusion-xl-1024-v1-0',
    'default_size': '1216x832',
    'default_quality': 'hd',
    'default_save_directory': '/custom/path/to/images'
}
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-provider`)
3. Commit changes (`git commit -m 'Add new AI provider'`)
4. Push to branch (`git push origin feature/new-provider`)
5. Open a Pull Request

**Development Setup:**
```bash
git clone https://github.com/YouvenZ/Imagegen_ink.git
cd Imagegen_ink
# Symlink for testing
ln -s $(pwd) ~/.config/inkscape/extensions/image_maker
```

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/YouvenZ/Imagegen_ink/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YouvenZ/Imagegen_ink/discussions)

---

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) - DALL-E API
- [Stability AI](https://stability.ai/) - Stable Diffusion API
- 