#!/usr/bin/env python3
"""
Inkscape extension to generate and edit images using AI providers.
Supports OpenAI DALL-E, Stability AI, Replicate, Venice AI, and local models.
"""

import inkex
from inkex import Image, Group
import urllib.request
import urllib.parse
import json
import ssl
import base64
import os
import tempfile
import hashlib
import time
import certifi
from datetime import datetime
from io import BytesIO
from pathlib import Path


class AIImageGenerator(inkex.EffectExtension):
    """Extension to generate and edit images using AI."""
    
    # Configuration file paths
    CONFIG_FILENAME = 'config.json'
    HISTORY_FILENAME = 'ai_image_history.json'
    
    # Provider configurations
    PROVIDERS = {
        'openai': {
            'name': 'OpenAI DALL-E',
            'generate_url': 'https://api.openai.com/v1/images/generations',
            'edit_url': 'https://api.openai.com/v1/images/edits',
            'variation_url': 'https://api.openai.com/v1/images/variations',
            'env_key': 'OPENAI_API_KEY',
            'config_key': 'openai_api_key',
            'models': ['dall-e-3', 'dall-e-2', 'gpt-image-1'],
            'sizes': ['1024x1024', '1024x1792', '1792x1024', '512x512', '256x256']
        },
        'stability': {
            'name': 'Stability AI',
            'generate_url': 'https://api.stability.ai/v1/generation/{engine}/text-to-image',
            'img2img_url': 'https://api.stability.ai/v1/generation/{engine}/image-to-image',
            'env_key': 'STABILITY_API_KEY',
            'config_key': 'stability_api_key',
            'models': ['stable-diffusion-xl-1024-v1-0', 'stable-diffusion-v1-6', 'stable-diffusion-xl-beta-v2-2-2'],
            'sizes': ['1024x1024', '1152x896', '896x1152', '1216x832', '832x1216', '512x512']
        },
        'replicate': {
            'name': 'Replicate',
            'generate_url': 'https://api.replicate.com/v1/predictions',
            'env_key': 'REPLICATE_API_TOKEN',
            'config_key': 'replicate_api_key',
            'models': ['stability-ai/sdxl', 'black-forest-labs/flux-schnell', 'black-forest-labs/flux-pro'],
            'sizes': ['1024x1024', '1024x768', '768x1024', '512x512']
        },
        'venice': {
            'name': 'Venice AI (experimental)',
            'generate_url': 'https://api.venice.ai/api/v1/image/generate',
            'edit_url': 'https://api.venice.ai/api/v1/image/edit',
            'img2img_url': 'https://api.venice.ai/api/v1/image/edit',
            'multi_edit_url': 'https://api.venice.ai/api/v1/image/multi-edit',
            'env_key': 'VENICE_API_KEY',
            'config_key': 'venice_api_key',
            'models': ['venice-sd35', 'z-image-turbo', 'qwen-image-3', 'flux-2-pro',
                       'nano-banana-2', 'seedream-v5-lite'],
            'sizes': ['1024x1024', '1280x720', '720x1280', '1024x768', '768x1024', '512x512']
        },
        'local': {
            'name': 'Local (Automatic1111/ComfyUI)',
            'generate_url': 'http://127.0.0.1:7860/sdapi/v1/txt2img',
            'img2img_url': 'http://127.0.0.1:7860/sdapi/v1/img2img',
            'env_key': '',
            'config_key': '',
            'models': ['default'],
            'sizes': ['1024x1024', '768x768', '512x512', '768x512', '512x768']
        }
    }
    
    # Venice generation models.
    # 'sizing' is 'px' for models that take width/height (widthHeightDivisor applies)
    # and 'ar' for models that require aspect_ratio and reject width/height.
    # Limits mirror GET https://api.venice.ai/api/v1/models?type=image
    VENICE_MODELS = {
        'venice-sd35': {'sizing': 'px', 'divisor': 16, 'max_steps': 30, 'prompt_limit': 1500},
        'z-image-turbo': {'sizing': 'px', 'divisor': 8, 'max_steps': 8, 'prompt_limit': 7500},
        'qwen-image-3': {'sizing': 'ar', 'max_steps': 50, 'prompt_limit': 10000},
        'flux-2-pro': {'sizing': 'ar', 'max_steps': 50, 'prompt_limit': 3000},
        'nano-banana-2': {'sizing': 'ar', 'max_steps': 50, 'prompt_limit': 32768},
        'seedream-v5-lite': {'sizing': 'ar', 'max_steps': 50, 'prompt_limit': 10000}
    }
    
    # Venice edit models (GET /models?type=inpaint). Used by edit and img2img.
    VENICE_EDIT_MODELS = {
        'firered-image-edit': {'prompt_limit': 1500},
        'qwen-image-3-edit': {'prompt_limit': 10000},
        'nano-banana-2-edit': {'prompt_limit': 32768}
    }
    
    VENICE_DEFAULT_MODEL = 'venice-sd35'
    VENICE_DEFAULT_EDIT_MODEL = 'firered-image-edit'
    
    # Venice has no mask channel, so the region is described in words instead.
    # Without this the model composes for the whole frame and the local composite
    # cuts whatever strays outside the mask.
    VENICE_MASK_HINTS = {
        'center': 'in the centre of the image',
        'edges': 'around the outer edges of the image, leaving the centre unchanged',
        'top_half': 'in the top half of the image',
        'bottom_half': 'in the bottom half of the image',
        'left_half': 'in the left half of the image',
        'right_half': 'in the right half of the image'
    }
    
    # Above this share of the canvas, naming a region misleads rather than helps
    VENICE_MASK_HINT_MAX_COVERAGE = 0.6
    
    # Aspect ratios accepted by every Venice model listed above
    VENICE_ASPECT_RATIOS = ('1:1', '3:2', '16:9', '21:9', '9:16', '2:3', '3:4', '4:5')
    
    # Venice API limits
    VENICE_MAX_DIMENSION = 1280
    VENICE_MAX_NEGATIVE_PROMPT = 7500
    VENICE_SEED_LIMIT = 999999999
    VENICE_MIN_IMAGE_PIXELS = 65536
    VENICE_MAX_IMAGE_PIXELS = 33177600
    VENICE_MAX_IMAGE_BYTES = 25 * 1024 * 1024
    
    # Preset configurations
    PRESETS = {
        'photorealistic': {
            'style': 'natural',
            'quality': 'hd',
            'negative_prompt': 'cartoon, illustration, painting, drawing, art, anime'
        },
        'artistic': {
            'style': 'vivid',
            'quality': 'hd',
            'negative_prompt': 'photo, realistic, photograph'
        },
        'quick_draft': {
            'style': 'natural',
            'quality': 'standard',
            'negative_prompt': ''
        },
        'high_quality': {
            'style': 'vivid',
            'quality': 'hd',
            'negative_prompt': 'low quality, blurry, distorted'
        }
    }
    
    def __init__(self):
        super().__init__()
        # Set config paths - extension directory for portability
        self.extension_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.extension_dir, self.CONFIG_FILENAME)
        self.history_path = os.path.join(self.extension_dir, self.HISTORY_FILENAME)
        
        # Load configuration on init
        self._config = self.load_config()
    
    def add_arguments(self, pars):
        pars.add_argument("--tab", type=str, default="mode", help="Active tab")
        pars.add_argument("--operation_mode", type=str, default="generate", help="Operation mode")
        
        # Provider settings
        pars.add_argument("--provider", type=str, default="openai", 
            help="AI Provider: openai, stability, replicate, local")
        pars.add_argument("--api_key", type=str, default="", help="API key (overrides config)")
        pars.add_argument("--use_env_key", type=inkex.Boolean, default=False,
            help="Use API key from environment variable")
        pars.add_argument("--use_config_key", type=inkex.Boolean, default=True,
            help="Use API key from config file")
        pars.add_argument("--api_endpoint", type=str, default="",
            help="Custom API endpoint for local/self-hosted models")
        pars.add_argument("--save_api_key", type=inkex.Boolean, default=False,
            help="Save provided API key to config file")
        
        # Proxy settings
        pars.add_argument("--use_proxy", type=inkex.Boolean, default=False, help="Use proxy")
        pars.add_argument("--proxy_url", type=str, default="", help="HTTP proxy URL")
        
        # Prompt settings
        pars.add_argument("--prompt", type=str, default="", help="Image description")
        pars.add_argument("--negative_prompt", type=str, default="", 
            help="What to avoid in generation")
        pars.add_argument("--edit_instruction", type=str, default="", help="Edit instruction")
        
        # Preset
        pars.add_argument("--preset", type=str, default="", 
            help="Load settings from preset: photorealistic, artistic, quick_draft, high_quality")
        
        # Image settings
        pars.add_argument("--model", type=str, default="dall-e-3", help="Model to use")
        pars.add_argument("--image_size", type=str, default="1024x1024", help="Image size")
        pars.add_argument("--custom_width", type=int, default=1024, help="Custom width")
        pars.add_argument("--custom_height", type=int, default=1024, help="Custom height")
        pars.add_argument("--use_custom_size", type=inkex.Boolean, default=False, 
            help="Use custom dimensions")
        pars.add_argument("--quality", type=str, default="standard", help="Image quality")
        pars.add_argument("--style", type=str, default="vivid", help="Image style")
        
        # Advanced generation options
        pars.add_argument("--seed", type=int, default=-1, 
            help="Random seed (-1 for random)")
        pars.add_argument("--batch_count", type=int, default=1, 
            help="Number of images to generate (1-4)")
        pars.add_argument("--cfg_scale", type=float, default=7.0,
            help="CFG scale for Stability/Local (1-20)")
        pars.add_argument("--steps", type=int, default=30,
            help="Sampling steps for Stability/Local")
        
        # Image-to-image options
        pars.add_argument("--img2img_strength", type=float, default=0.75,
            help="How much to transform source image (0-1)")
        pars.add_argument("--use_selection_as_mask", type=inkex.Boolean, default=False,
            help="Use selected shapes as edit mask")
        
        # Venice options
        pars.add_argument("--hide_watermark", type=inkex.Boolean, default=True,
            help="Ask Venice not to watermark generated images")
        
        # Save options
        pars.add_argument("--save_to_disk", type=inkex.Boolean, default=True, help="Save to disk")
        pars.add_argument("--save_directory", type=str, default="", help="Save directory")
        pars.add_argument("--filename_prefix", type=str, default="ai_image", help="Filename prefix")
        pars.add_argument("--embed_in_svg", type=inkex.Boolean, default=True, help="Embed in SVG")
        
        # Placement options
        pars.add_argument("--position_mode", type=str, default="center", help="Position mode")
        pars.add_argument("--scale_mode", type=str, default="original", help="Scale mode")
        pars.add_argument("--placement_width", type=int, default=800, help="Placement width")
        pars.add_argument("--placement_height", type=int, default=600, help="Placement height")
        
        # Edit options
        pars.add_argument("--mask_mode", type=str, default="full", help="Mask mode for editing")
        pars.add_argument("--mask_opacity", type=float, default=0.5, help="Mask opacity")
        pars.add_argument("--mask_feather", type=int, default=0, help="Mask feather radius")
        
        # History/Cache
        pars.add_argument("--save_history", type=inkex.Boolean, default=True,
            help="Save generation history")
    
    # ==================== Configuration Management ====================
    
    def load_config(self):
        """Load configuration from JSON file."""
        default_config = {
            'openai_api_key': '',
            'stability_api_key': '',
            'replicate_api_key': '',
            'venice_api_key': '',
            'default_provider': 'openai',
            'default_model': 'dall-e-3',
            'default_size': '1024x1024',
            'default_quality': 'standard',
            'default_save_directory': os.path.expanduser('~/Pictures/AI_Images')
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    default_config.update(loaded_config)
            except Exception as e:
                inkex.errormsg(f"Warning: Could not load config file: {e}")
        
        return default_config
    
    def save_config(self, config=None):
        """Save configuration to JSON file."""
        if config is None:
            config = self._config
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            inkex.errormsg(f"Warning: Could not save config file: {e}")
    
    def get_config_value(self, key, default=None):
        """Get a value from the configuration."""
        return self._config.get(key, default)
    
    def set_config_value(self, key, value):
        """Set a value in the configuration and save."""
        self._config[key] = value
        self.save_config()
    
    def get_api_key(self):
        """
        Get API key with priority:
        1. Direct input (if provided and not placeholder)
        2. Environment variable (if use_env_key is True)
        3. Config file (if use_config_key is True)
        """
        provider = self.options.provider
        
        # Skip API key for local provider
        if provider == 'local':
            return ''
        
        # 1. Check direct input first
        if self.options.api_key and self.options.api_key not in ['', 'sk-...', 'sk-your-key-here']:
            # Save to config if requested
            if self.options.save_api_key:
                config_key = self.PROVIDERS.get(provider, {}).get('config_key', '')
                if config_key:
                    self.set_config_value(config_key, self.options.api_key)
            return self.options.api_key
        
        # 2. Check environment variable
        if self.options.use_env_key:
            env_key = self.PROVIDERS.get(provider, {}).get('env_key', '')
            if env_key:
                env_value = os.environ.get(env_key, '')
                if env_value:
                    return env_value
        
        # 3. Check config file
        if self.options.use_config_key:
            config_key = self.PROVIDERS.get(provider, {}).get('config_key', '')
            if config_key:
                config_value = self.get_config_value(config_key, '')
                if config_value and config_value not in ['sk-your-key-here', 'r8_your-token-here']:
                    return config_value
        
        return ''
    
    def get_save_directory(self):
        """Get save directory from options or config."""
        if self.options.save_directory and self.options.save_directory.strip():
            return self.options.save_directory
        return self.get_config_value('default_save_directory', os.path.expanduser('~/Pictures/AI_Images'))
    
    # ==================== History Management ====================
    
    def load_history(self):
        """Load generation history."""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def save_to_history(self, operation, prompt):
        """Save operation to history file."""
        history = self.load_history()
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'prompt': prompt,
            'provider': self.options.provider,
            'model': self.options.model,
            'size': self.options.image_size,
            'seed': self.options.seed if self.options.seed != -1 else 'random'
        })
        
        # Keep only last 100 entries
        history = history[-100:]
        
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
        except:
            pass
    
    # ==================== Main Effect ====================
    
    def effect(self):
        """Main effect function."""
        # Apply preset if specified
        self.apply_preset()
        
        # Apply config defaults if not overridden
        self.apply_config_defaults()
        
        # Get API key (from input, env, or config)
        api_key = self.get_api_key()
        if not api_key and self.options.provider != 'local':
            provider_info = self.PROVIDERS.get(self.options.provider, {})
            inkex.errormsg(
                f"No API key found for {provider_info.get('name', self.options.provider)}.\n\n"
                f"You can provide an API key in one of these ways:\n"
                f"1. Enter it directly in the API Key field\n"
                f"2. Set environment variable: {provider_info.get('env_key', 'N/A')}\n"
                f"3. Add it to the config file: {self.config_path}\n\n"
                f"Config file format:\n"
                f'{{\n'
                f'    "{provider_info.get("config_key", "api_key")}": "your-api-key-here"\n'
                f'}}'
            )
            return
        
        # Store resolved API key
        self._api_key = api_key
        
        # Handle different operation modes
        if self.options.operation_mode == "generate":
            self.handle_generate()
        elif self.options.operation_mode == "edit":
            self.handle_edit()
        elif self.options.operation_mode == "variation":
            self.handle_variation()
        elif self.options.operation_mode == "img2img":
            self.handle_img2img()
    
    def apply_preset(self):
        """Apply preset settings if specified."""
        if self.options.preset and self.options.preset in self.PRESETS:
            preset = self.PRESETS[self.options.preset]
            if not self.options.style:
                self.options.style = preset.get('style', 'vivid')
            if not self.options.quality:
                self.options.quality = preset.get('quality', 'standard')
            if not self.options.negative_prompt:
                self.options.negative_prompt = preset.get('negative_prompt', '')
    
    def apply_config_defaults(self):
        """Apply defaults from config file if options not explicitly set."""
        # Use config defaults for model if using default
        if self.options.model == 'dall-e-3':
            default_model = self.get_config_value('default_model', 'dall-e-3')
            # Only override if provider matches
            if self.options.provider == 'openai' and default_model.startswith('dall-e'):
                self.options.model = default_model
        
        # A Venice run still carrying the model dropdown's OpenAI default gets a
        # Venice model instead, so the Model tab does not have to be touched first
        if self.options.provider == 'venice' and self.options.model == 'dall-e-3':
            default_model = self.get_config_value('default_model', self.VENICE_DEFAULT_MODEL)
            if default_model in self.VENICE_MODELS or default_model in self.VENICE_EDIT_MODELS:
                self.options.model = default_model
            else:
                self.options.model = self.VENICE_DEFAULT_MODEL
        
        # Use config defaults for size
        if self.options.image_size == '1024x1024':
            self.options.image_size = self.get_config_value('default_size', '1024x1024')
        
        # Use config defaults for quality
        if self.options.quality == 'standard':
            self.options.quality = self.get_config_value('default_quality', 'standard')
    
    # ==================== Operation Handlers ====================
    
    def handle_generate(self):
        """Handle image generation."""
        if not self.options.prompt or len(self.options.prompt.strip()) < 3:
            inkex.errormsg("Please provide a description for image generation.")
            return
        
        # Generate images (batch support)
        batch_count = max(1, min(4, self.options.batch_count))
        
        for i in range(batch_count):
            image_data = self.generate_image()
            if image_data:
                # Offset position for batch images
                offset = i * 50 if batch_count > 1 else 0
                self.add_image_to_document(image_data, offset=offset)
                
                # Save to history
                if self.options.save_history:
                    self.save_to_history('generate', self.options.prompt)
    
    def handle_edit(self):
        """Handle image editing."""
        selected_image = self.get_selected_image()
        if not selected_image:
            inkex.errormsg("Please select an image to edit.")
            return
        
        if not self.options.edit_instruction or len(self.options.edit_instruction.strip()) < 3:
            inkex.errormsg("Please provide edit instructions.")
            return
        
        image_data = self.edit_image(selected_image)
        if image_data:
            self.replace_image(selected_image['element'], image_data)
            
            if self.options.save_history:
                self.save_to_history('edit', self.options.edit_instruction)
    
    def handle_variation(self):
        """Handle creating variations."""
        selected_image = self.get_selected_image()
        if not selected_image:
            inkex.errormsg("Please select an image to create a variation of.")
            return
        
        image_data = self.create_variation(selected_image)
        if image_data:
            self.add_image_to_document(image_data)
            
            if self.options.save_history:
                self.save_to_history('variation', 'Created variation')
    
    def handle_img2img(self):
        """Handle image-to-image transformation."""
        selected_image = self.get_selected_image()
        if not selected_image:
            inkex.errormsg("Please select an image for img2img transformation.")
            return
        
        if not self.options.prompt or len(self.options.prompt.strip()) < 3:
            inkex.errormsg("Please provide a prompt for img2img transformation.")
            return
        
        image_data = self.img2img(selected_image)
        if image_data:
            self.add_image_to_document(image_data)
            
            if self.options.save_history:
                self.save_to_history('img2img', self.options.prompt)
    
    # ==================== Image Selection ====================
    
    def get_selected_image(self):
        """Get selected image element."""
        if not self.svg.selection:
            return None
        
        for elem in self.svg.selection:
            if isinstance(elem, Image):
                return {
                    'element': elem,
                    'href': elem.get('xlink:href') or elem.get('href')
                }
            
            elif isinstance(elem, Group):
                for child in elem:
                    if isinstance(child, Image):
                        return {
                            'element': child,
                            'href': child.get('xlink:href') or child.get('href')
                        }
        
        return None
    
    def get_selected_shapes_as_mask(self):
        """Get selected shapes to use as mask."""
        if not self.options.use_selection_as_mask:
            return None
        
        shapes = []
        for elem in self.svg.selection:
            if not isinstance(elem, Image):
                shapes.append(elem)
        
        return shapes if shapes else None
    
    def get_image_size(self):
        """Get image size, supporting custom dimensions."""
        if self.options.use_custom_size:
            return f"{self.options.custom_width}x{self.options.custom_height}"
        return self.options.image_size
    
    # ==================== Image Generation ====================
    
    def generate_image(self):
        """Generate new image using selected provider."""
        provider = self.options.provider
        
        if provider == 'openai':
            return self.generate_openai()
        elif provider == 'stability':
            return self.generate_stability()
        elif provider == 'replicate':
            return self.generate_replicate()
        elif provider == 'venice':
            return self.generate_venice()
        elif provider == 'local':
            return self.generate_local()
        else:
            inkex.errormsg(f"Unknown provider: {provider}")
            return None
    
    def generate_openai(self):
        """Generate image using OpenAI DALL-E."""
        url = self.PROVIDERS['openai']['generate_url']
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}'
        }
        
        data = {
            'model': self.options.model,
            'prompt': self.build_prompt(),
            'n': 1,
            'size': self.get_image_size(),
            'response_format': 'b64_json'  # Get base64 directly
        }
        
        # Add DALL-E 3 specific parameters
        if self.options.model == 'dall-e-3':
            data['quality'] = self.options.quality
            data['style'] = self.options.style
        
        result = self.call_api(url, headers, data)
        if result and 'data' in result and len(result['data']) > 0:
            if 'b64_json' in result['data'][0]:
                return base64.b64decode(result['data'][0]['b64_json'])
            elif 'url' in result['data'][0]:
                return self.download_image(result['data'][0]['url'])
        return None
    
    def generate_stability(self):
        """Generate image using Stability AI."""
        engine = self.options.model or 'stable-diffusion-xl-1024-v1-0'
        url = self.PROVIDERS['stability']['generate_url'].format(engine=engine)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
            'Accept': 'application/json'
        }
        
        width, height = map(int, self.get_image_size().split('x'))
        
        data = {
            'text_prompts': [
                {'text': self.options.prompt, 'weight': 1.0}
            ],
            'cfg_scale': self.options.cfg_scale,
            'steps': self.options.steps,
            'width': width,
            'height': height,
            'samples': 1
        }
        
        # Add negative prompt if provided
        if self.options.negative_prompt:
            data['text_prompts'].append({
                'text': self.options.negative_prompt,
                'weight': -1.0
            })
        
        # Add seed if specified
        if self.options.seed != -1:
            data['seed'] = self.options.seed
        
        result = self.call_api(url, headers, data)
        if result and 'artifacts' in result and len(result['artifacts']) > 0:
            return base64.b64decode(result['artifacts'][0]['base64'])
        return None
    
    def generate_replicate(self):
        """Generate image using Replicate."""
        url = self.PROVIDERS['replicate']['generate_url']
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self._api_key}'
        }
        
        width, height = map(int, self.get_image_size().split('x'))
        
        # Determine model version
        model = self.options.model or 'stability-ai/sdxl'
        
        data = {
            'version': self.get_replicate_version(model),
            'input': {
                'prompt': self.build_prompt(),
                'width': width,
                'height': height
            }
        }
        
        if self.options.negative_prompt:
            data['input']['negative_prompt'] = self.options.negative_prompt
        
        if self.options.seed != -1:
            data['input']['seed'] = self.options.seed
        
        # Start prediction
        result = self.call_api(url, headers, data)
        if not result or 'id' not in result:
            return None
        
        # Poll for completion
        prediction_id = result['id']
        return self.poll_replicate(prediction_id)
    
    def get_replicate_version(self, model):
        """Get Replicate model version."""
        versions = {
            'stability-ai/sdxl': 'da77bc59ee60423279fd632efb4795ab731d9e3ca9705ef3341091fb989b7eaf',
            'black-forest-labs/flux-schnell': 'f2ab8a5bfe79f02f0789a146cf5e73d2a4ff2684a98c2b303d1e1ff3814271db',
            'black-forest-labs/flux-pro': '4f6c0f2a74f7f5e43c6e2e3e3f0e8b6d2a4c8f0e2b4a6c8d0e2f4a6b8c0d2e4f6'
        }
        return versions.get(model, versions['stability-ai/sdxl'])
    
    def poll_replicate(self, prediction_id, max_attempts=60):
        """Poll Replicate for prediction completion."""
        url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
        
        headers = {
            'Authorization': f'Token {self._api_key}'
        }
        
        for _ in range(max_attempts):
            result = self.call_api_get(url, headers)
            if not result:
                return None
            
            status = result.get('status')
            
            if status == 'succeeded':
                output = result.get('output')
                if output:
                    # Output is usually a list of URLs
                    if isinstance(output, list) and len(output) > 0:
                        return self.download_image(output[0])
                    elif isinstance(output, str):
                        return self.download_image(output)
                return None
            
            elif status == 'failed':
                error = result.get('error', 'Unknown error')
                inkex.errormsg(f"Replicate prediction failed: {error}")
                return None
            
            # Still processing, wait and retry
            time.sleep(2)
        
        inkex.errormsg("Replicate prediction timed out")
        return None
    
    def generate_local(self):
        """Generate image using local Automatic1111/ComfyUI API."""
        endpoint = self.options.api_endpoint or self.PROVIDERS['local']['generate_url']
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        width, height = map(int, self.get_image_size().split('x'))
        
        data = {
            'prompt': self.build_prompt(),
            'negative_prompt': self.options.negative_prompt or '',
            'width': width,
            'height': height,
            'steps': self.options.steps,
            'cfg_scale': self.options.cfg_scale,
            'sampler_name': 'DPM++ 2M Karras',
            'batch_size': 1
        }
        
        if self.options.seed != -1:
            data['seed'] = self.options.seed
        
        result = self.call_api(endpoint, headers, data, use_ssl=False)
        if result and 'images' in result and len(result['images']) > 0:
            return base64.b64decode(result['images'][0])
        return None
    
    def generate_venice(self):
        """Generate image using Venice AI (experimental)."""
        url = self.PROVIDERS['venice']['generate_url']
        model = self.get_venice_model()
        
        if not self.check_venice_prompt(self.options.prompt, model, self.VENICE_MODELS):
            return None
        
        if not self.check_venice_negative_prompt():
            return None
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}'
        }
        
        data = {
            'model': model,
            'prompt': self.options.prompt,
            'format': 'png',
            'cfg_scale': self.options.cfg_scale,
            'steps': self.get_venice_steps(model),
            'hide_watermark': self.options.hide_watermark
        }
        
        # Venice takes a real negative prompt, so no folding into the prompt text
        if self.options.negative_prompt:
            data['negative_prompt'] = self.options.negative_prompt
        
        seed = self.get_venice_seed()
        if seed is not None:
            data['seed'] = seed
        
        data.update(self.get_venice_size(model))
        
        result = self.call_api(url, headers, data)
        if result and 'images' in result and len(result['images']) > 0:
            image_data = base64.b64decode(result['images'][0])
            self.sync_placement_size(image_data)
            return image_data
        return None
    
    def get_venice_model(self):
        """Resolve the selected model to a Venice generation model."""
        model = self.options.model
        
        if model in self.VENICE_MODELS:
            return model
        
        inkex.errormsg(
            f"Note: '{model}' is not a Venice generation model, "
            f"so '{self.VENICE_DEFAULT_MODEL}' was used instead.\n"
            f"Venice models in the Model list: {', '.join(self.VENICE_MODELS)}"
        )
        return self.VENICE_DEFAULT_MODEL
    
    def get_venice_edit_model(self):
        """Resolve the selected model to a Venice edit model."""
        model = self.options.model
        
        if model in self.VENICE_EDIT_MODELS:
            return model
        
        inkex.errormsg(
            f"Note: '{model}' cannot edit images, "
            f"so '{self.VENICE_DEFAULT_EDIT_MODEL}' was used instead.\n"
            f"Venice edit models in the Model list: {', '.join(self.VENICE_EDIT_MODELS)}"
        )
        return self.VENICE_DEFAULT_EDIT_MODEL
    
    def check_venice_prompt(self, prompt, model, models):
        """Check a prompt against the Venice model's character limit."""
        if not prompt:
            return True
        
        limit = models.get(model, {}).get('prompt_limit', 1500)
        if len(prompt) > limit:
            inkex.errormsg(
                f"Prompt is {len(prompt)} characters but Venice model '{model}' "
                f"accepts at most {limit}. Shorten the prompt or pick another model."
            )
            return False
        return True
    
    def check_venice_negative_prompt(self):
        """Check the negative prompt against Venice's limit."""
        if len(self.options.negative_prompt or '') > self.VENICE_MAX_NEGATIVE_PROMPT:
            inkex.errormsg(
                f"Negative prompt is longer than Venice's "
                f"{self.VENICE_MAX_NEGATIVE_PROMPT} character limit."
            )
            return False
        return True
    
    def get_venice_steps(self, model):
        """Clamp sampling steps to what the Venice model accepts."""
        max_steps = self.VENICE_MODELS.get(model, {}).get('max_steps', 30)
        
        if self.options.steps > max_steps:
            inkex.errormsg(
                f"Note: Venice model '{model}' accepts at most {max_steps} sampling "
                f"steps, so {self.options.steps} was reduced to {max_steps}."
            )
            return max_steps
        return self.options.steps
    
    def get_venice_seed(self):
        """Get the seed in Venice's accepted range, or None for a random seed."""
        if self.options.seed == -1:
            return None
        
        if abs(self.options.seed) > self.VENICE_SEED_LIMIT:
            inkex.errormsg(
                f"Note: Venice seeds range from -{self.VENICE_SEED_LIMIT} to "
                f"{self.VENICE_SEED_LIMIT}, so {self.options.seed} was clamped."
            )
            return max(-self.VENICE_SEED_LIMIT,
                       min(self.VENICE_SEED_LIMIT, self.options.seed))
        return self.options.seed
    
    def get_venice_size(self, model):
        """Build the size fields for a Venice model.
        
        Venice image models come in two families: pixel based models take width and
        height, while the rest reject them and take an aspect_ratio instead.
        """
        if self.VENICE_MODELS.get(model, {}).get('sizing') == 'px':
            divisor = self.VENICE_MODELS[model].get('divisor', 8)
            width, height = self.get_venice_pixel_size(divisor)
            return {'width': width, 'height': height}
        
        return {'aspect_ratio': self.get_venice_aspect_ratio()}
    
    def get_venice_pixel_size(self, divisor):
        """Fit the selected size into Venice's pixel limits for the model."""
        width, height = self.parse_image_size()
        
        # Scale down together so the requested proportions survive the cap
        largest = max(width, height)
        if largest > self.VENICE_MAX_DIMENSION:
            scale = self.VENICE_MAX_DIMENSION / largest
            width = int(width * scale)
            height = int(height * scale)
        
        return (self.align_to_divisor(width, divisor),
                self.align_to_divisor(height, divisor))
    
    def align_to_divisor(self, value, divisor):
        """Round a dimension to the nearest multiple the model accepts."""
        aligned = int(round(value / divisor)) * divisor
        return max(divisor, min(self.VENICE_MAX_DIMENSION, aligned))
    
    def get_venice_aspect_ratio(self):
        """Pick the supported aspect ratio closest to the selected size."""
        width, height = self.parse_image_size()
        target = width / height
        
        best = self.VENICE_ASPECT_RATIOS[0]
        best_distance = None
        
        for ratio in self.VENICE_ASPECT_RATIOS:
            ratio_width, ratio_height = map(int, ratio.split(':'))
            distance = abs(ratio_width / ratio_height - target)
            if best_distance is None or distance < best_distance:
                best = ratio
                best_distance = distance
        
        return best
    
    def parse_image_size(self):
        """Parse the selected size string, falling back to a square."""
        try:
            width, height = map(int, self.get_image_size().split('x'))
            if width > 0 and height > 0:
                return width, height
        except (ValueError, AttributeError):
            pass
        return 1024, 1024
    
    def get_png_size(self, image_data):
        """Read a PNG's dimensions from its IHDR chunk, without Pillow."""
        if len(image_data) < 24 or image_data[:8] != b'\x89PNG\r\n\x1a\n':
            return None
        
        return (int.from_bytes(image_data[16:20], 'big'),
                int.from_bytes(image_data[20:24], 'big'))
    
    def sync_placement_size(self, image_data):
        """Match the placement size to the image that actually came back.
        
        Venice models that take an aspect_ratio rather than exact pixels can answer
        with different dimensions than the Size dropdown asked for. Without this the
        image would be stretched into the requested box.
        """
        size = self.get_png_size(image_data)
        if not size or size == self.parse_image_size():
            return
        
        self.options.use_custom_size = True
        self.options.custom_width, self.options.custom_height = size
    
    def build_prompt(self):
        """Build full prompt with any modifications."""
        prompt = self.options.prompt
        
        # For providers that don't support negative prompts in API,
        # we could append style hints to main prompt
        if self.options.provider == 'openai' and self.options.negative_prompt:
            # DALL-E doesn't support negative prompts directly
            # But we can hint at what we don't want
            prompt += f". Avoid: {self.options.negative_prompt}"
        
        return prompt
    
    # ==================== Image Editing ====================
    
    def convert_image_to_rgba(self, image_data):
        """Convert image to RGBA format required by DALL-E."""
        try:
            from PIL import Image as PILImage
            
            img = PILImage.open(BytesIO(image_data))
            
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Get target size
            size_str = self.get_image_size()
            size = int(size_str.split('x')[0])
            if size not in [256, 512, 1024]:
                size = 1024
            
            if img.size != (size, size):
                img.thumbnail((size, size), PILImage.Resampling.LANCZOS)
                
                new_img = PILImage.new('RGBA', (size, size), (0, 0, 0, 0))
                x = (size - img.size[0]) // 2
                y = (size - img.size[1]) // 2
                new_img.paste(img, (x, y))
                img = new_img
            
            output = BytesIO()
            img.save(output, format='PNG')
            return output.getvalue()
            
        except ImportError:
            inkex.errormsg("PIL/Pillow library required for image editing. Install with: pip install Pillow")
            return None
        except Exception as e:
            inkex.errormsg(f"Error converting image: {str(e)}")
            return None
    
    def create_mask(self, image_data, mask_mode='full'):
        """Create a mask for image editing."""
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFilter
            
            img = PILImage.open(BytesIO(image_data))
            size = img.size[0]
            
            if mask_mode == 'full':
                mask = PILImage.new('RGBA', (size, size), (0, 0, 0, 0))
            
            elif mask_mode == 'center':
                mask = PILImage.new('RGBA', (size, size), (0, 0, 0, 255))
                draw = ImageDraw.Draw(mask)
                margin = size // 4
                draw.rectangle(
                    [margin, margin, size - margin, size - margin],
                    fill=(0, 0, 0, 0)
                )
            
            elif mask_mode == 'edges':
                mask = PILImage.new('RGBA', (size, size), (0, 0, 0, 0))
                draw = ImageDraw.Draw(mask)
                margin = size // 4
                draw.rectangle(
                    [margin, margin, size - margin, size - margin],
                    fill=(0, 0, 0, 255)
                )
            
            elif mask_mode == 'top_half':
                mask = PILImage.new('RGBA', (size, size), (0, 0, 0, 255))
                draw = ImageDraw.Draw(mask)
                draw.rectangle([0, 0, size, size // 2], fill=(0, 0, 0, 0))
            
            elif mask_mode == 'bottom_half':
                mask = PILImage.new('RGBA', (size, size), (0, 0, 0, 255))
                draw = ImageDraw.Draw(mask)
                draw.rectangle([0, size // 2, size, size], fill=(0, 0, 0, 0))
            
            elif mask_mode == 'left_half':
                mask = PILImage.new('RGBA', (size, size), (0, 0, 0, 255))
                draw = ImageDraw.Draw(mask)
                draw.rectangle([0, 0, size // 2, size], fill=(0, 0, 0, 0))
            
            elif mask_mode == 'right_half':
                mask = PILImage.new('RGBA', (size, size), (0, 0, 0, 255))
                draw = ImageDraw.Draw(mask)
                draw.rectangle([size // 2, 0, size, size], fill=(0, 0, 0, 0))
            
            else:
                mask = PILImage.new('RGBA', (size, size), (0, 0, 0, 0))
            
            # Apply feathering if specified
            if self.options.mask_feather > 0:
                # Convert to grayscale for blur, then back
                alpha = mask.split()[3]
                alpha = alpha.filter(ImageFilter.GaussianBlur(self.options.mask_feather))
                mask.putalpha(alpha)
            
            output = BytesIO()
            mask.save(output, format='PNG')
            return output.getvalue()
            
        except ImportError:
            inkex.errormsg("PIL/Pillow library required. Install with: pip install Pillow")
            return None
        except Exception as e:
            inkex.errormsg(f"Error creating mask: {str(e)}")
            return None
    
    def create_mask_from_shapes(self, image_data, shapes):
        """Create mask from selected Inkscape shapes."""
        try:
            from PIL import Image as PILImage, ImageDraw
            
            img = PILImage.open(BytesIO(image_data))
            size = img.size[0]
            
            # Create opaque mask (keep everything by default)
            mask = PILImage.new('RGBA', (size, size), (0, 0, 0, 255))
            draw = ImageDraw.Draw(mask)
            
            for shape in shapes:
                bbox = shape.bounding_box()
                if bbox:
                    # Convert to mask coordinates (simplified)
                    x1 = int(bbox.left * size / self.svg.viewport_width)
                    y1 = int(bbox.top * size / self.svg.viewport_height)
                    x2 = int(bbox.right * size / self.svg.viewport_width)
                    y2 = int(bbox.bottom * size / self.svg.viewport_height)
                    
                    # Make this area transparent (to be regenerated)
                    draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 0))
            
            output = BytesIO()
            mask.save(output, format='PNG')
            return output.getvalue()
            
        except Exception as e:
            inkex.errormsg(f"Error creating mask from shapes: {str(e)}")
            return None
    
    def edit_image(self, selected_image):
        """Edit existing image."""
        image_data = self.get_image_data(selected_image['href'])
        if not image_data:
            inkex.errormsg("Could not load image data for editing.")
            return None
        
        # Venice keeps the original framing. It has no mask channel and infers the
        # output aspect ratio from the input image, so the transparent square padding
        # added for DALL-E would be baked into the result.
        if self.options.provider == 'venice':
            return self.edit_venice(image_data)
        
        image_data = self.convert_image_to_rgba(image_data)
        if not image_data:
            return None
        
        # Create mask - either from shapes or from mask_mode
        shapes = self.get_selected_shapes_as_mask()
        if shapes:
            mask_data = self.create_mask_from_shapes(image_data, shapes)
        else:
            mask_data = self.create_mask(image_data, self.options.mask_mode)
        
        if not mask_data:
            return None
        
        if self.options.provider == 'openai':
            return self.edit_openai(image_data, mask_data)
        elif self.options.provider == 'stability':
            return self.edit_stability(image_data, mask_data)
        elif self.options.provider == 'local':
            return self.edit_local(image_data, mask_data)
        else:
            inkex.errormsg(f"Edit not supported for provider: {self.options.provider}")
            return None
    
    def edit_openai(self, image_data, mask_data):
        """Edit image using OpenAI API."""
        url = self.PROVIDERS['openai']['edit_url']
        
        headers = {
            'Authorization': f'Bearer {self._api_key}'
        }
        
        boundary = '----WebKitFormBoundary' + os.urandom(16).hex()
        headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
        
        body_parts = []
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="image"; filename="image.png"')
        body_parts.append(b'Content-Type: image/png')
        body_parts.append(b'')
        body_parts.append(image_data)
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="mask"; filename="mask.png"')
        body_parts.append(b'Content-Type: image/png')
        body_parts.append(b'')
        body_parts.append(mask_data)
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="prompt"')
        body_parts.append(b'')
        body_parts.append(self.options.edit_instruction.encode())
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="model"')
        body_parts.append(b'')
        body_parts.append(b'dall-e-2')
        
        size = self.get_image_size()
        if size not in ['256x256', '512x512', '1024x1024']:
            size = '1024x1024'
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="size"')
        body_parts.append(b'')
        body_parts.append(size.encode())
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="response_format"')
        body_parts.append(b'')
        body_parts.append(b'b64_json')
        
        body_parts.append(f'--{boundary}--'.encode())
        body_parts.append(b'')
        
        body = b'\r\n'.join(body_parts)
        
        result = self.call_api_multipart(url, headers, body)
        if result and 'data' in result and len(result['data']) > 0:
            if 'b64_json' in result['data'][0]:
                return base64.b64decode(result['data'][0]['b64_json'])
            elif 'url' in result['data'][0]:
                return self.download_image(result['data'][0]['url'])
        return None
    
    def edit_stability(self, image_data, mask_data):
        """Edit image using Stability AI inpainting."""
        engine = 'stable-inpainting-512-v2-0'
        url = f"https://api.stability.ai/v1/generation/{engine}/image-to-image/masking"
        
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Accept': 'application/json'
        }
        
        boundary = '----WebKitFormBoundary' + os.urandom(16).hex()
        headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
        
        body_parts = []
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="init_image"; filename="image.png"')
        body_parts.append(b'Content-Type: image/png')
        body_parts.append(b'')
        body_parts.append(image_data)
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="mask_image"; filename="mask.png"')
        body_parts.append(b'Content-Type: image/png')
        body_parts.append(b'')
        body_parts.append(mask_data)
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="text_prompts[0][text]"')
        body_parts.append(b'')
        body_parts.append(self.options.edit_instruction.encode())
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="text_prompts[0][weight]"')
        body_parts.append(b'')
        body_parts.append(b'1.0')
        
        body_parts.append(f'--{boundary}--'.encode())
        body_parts.append(b'')
        
        body = b'\r\n'.join(body_parts)
        
        result = self.call_api_multipart(url, headers, body)
        if result and 'artifacts' in result and len(result['artifacts']) > 0:
            return base64.b64decode(result['artifacts'][0]['base64'])
        return None
    
    def edit_local(self, image_data, mask_data):
        """Edit image using local API inpainting."""
        endpoint = self.options.api_endpoint or 'http://127.0.0.1:7860/sdapi/v1/img2img'
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            'init_images': [base64.b64encode(image_data).decode('utf-8')],
            'mask': base64.b64encode(mask_data).decode('utf-8'),
            'prompt': self.options.edit_instruction,
            'negative_prompt': self.options.negative_prompt or '',
            'denoising_strength': self.options.img2img_strength,
            'steps': self.options.steps,
            'cfg_scale': self.options.cfg_scale,
            'inpainting_fill': 1,
            'inpaint_full_res': True
        }
        
        if self.options.seed != -1:
            data['seed'] = self.options.seed
        
        result = self.call_api(endpoint, headers, data, use_ssl=False)
        if result and 'images' in result and len(result['images']) > 0:
            return base64.b64decode(result['images'][0])
        return None
    
    def edit_venice(self, image_data):
        """Edit image using Venice (experimental).
        
        Venice has no mask channel - its edit endpoint rewrites the whole frame. When
        a partial mask is selected, the returned frame is composited back over the
        original through that mask locally, so only the masked region changes.
        """
        if not self.check_venice_image(image_data):
            return None
        
        instruction = self.add_venice_region_hint(self.options.edit_instruction)
        
        edited_data = self.request_venice_edit(image_data, instruction)
        if not edited_data:
            return None
        
        return self.apply_venice_mask(image_data, edited_data)
    
    def add_venice_region_hint(self, instruction):
        """Name the masked region in the instruction.
        
        Venice is never told where the edit belongs, so it composes for the whole
        frame and the local composite then keeps only the masked part. Naming the
        region in words keeps the subject inside it far more often.
        """
        if not instruction:
            return instruction
        
        shapes = self.get_selected_shapes_as_mask()
        
        if shapes:
            hint = self.describe_shapes_region(shapes)
        else:
            hint = self.VENICE_MASK_HINTS.get(self.options.mask_mode)
        
        if not hint:
            return instruction
        
        return f"{instruction.rstrip().rstrip('.')}, {hint}"
    
    def describe_shapes_region(self, shapes):
        """Describe where the selected shapes sit, as a position on a 3x3 grid.
        
        Returns None when the selection covers most of the image, where naming a
        region would point the model at the wrong place.
        """
        try:
            boxes = [shape.bounding_box() for shape in shapes]
            boxes = [box for box in boxes if box]
            
            width = self.svg.viewport_width
            height = self.svg.viewport_height
            
            if not boxes or not width or not height:
                return None
            
            # Combined extent of the selection, not the average of its parts
            left = min(box.left for box in boxes)
            top = min(box.top for box in boxes)
            right = max(box.right for box in boxes)
            bottom = max(box.bottom for box in boxes)
            
            covered = ((right - left) * (bottom - top)) / (width * height)
            if covered > self.VENICE_MASK_HINT_MAX_COVERAGE:
                return None
            
            columns = ('left', 'centre', 'right')
            rows = ('top', 'middle', 'bottom')
            
            column = columns[max(0, min(2, int((left + right) / 2 / width * 3)))]
            row = rows[max(0, min(2, int((top + bottom) / 2 / height * 3)))]
            
            if column == 'centre' and row == 'middle':
                return 'in the centre of the image'
            
            return f"in the {row} {column} area of the image"
        
        except Exception:
            return None
    
    def img2img_venice(self, image_data):
        """Image-to-image using Venice (experimental)."""
        if not self.check_venice_image(image_data):
            return None
        
        # Only mention the ignored control when the user actually changed it
        if abs(self.options.img2img_strength - 0.75) > 0.001:
            inkex.errormsg(
                "Note: Venice's edit endpoint has no transformation strength "
                "parameter, so 'Transformation strength' was ignored."
            )
        
        result = self.request_venice_edit(image_data, self.options.prompt)
        if result:
            # img2img places a new image, so the box has to match what came back
            self.sync_placement_size(result)
        return result
    
    def request_venice_edit(self, image_data, prompt):
        """Send an image and prompt to Venice, returning raw image bytes."""
        url = self.PROVIDERS['venice']['edit_url']
        model = self.get_venice_edit_model()
        
        if not prompt or len(prompt.strip()) < 3:
            inkex.errormsg("Please provide instructions describing the edit.")
            return None
        
        # The edit endpoint has no negative_prompt field, so hint in the prompt
        # instead - the same approach build_prompt() takes for DALL-E
        if self.options.negative_prompt:
            prompt += f". Avoid: {self.options.negative_prompt}"
        
        if not self.check_venice_prompt(prompt, model, self.VENICE_EDIT_MODELS):
            return None
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
            'Accept': 'image/png'
        }
        
        # aspect_ratio is left out so Venice infers it from the input image.
        # hide_watermark exists only on /image/generate, so edits cannot suppress it.
        data = {
            'model': model,
            'prompt': prompt,
            'image': base64.b64encode(image_data).decode('utf-8'),
            'output_format': 'png'
        }
        
        return self.call_api_binary(url, headers, data)
    
    def check_venice_image(self, image_data):
        """Check an input image against Venice's edit endpoint limits."""
        if len(image_data) >= self.VENICE_MAX_IMAGE_BYTES:
            inkex.errormsg(
                f"Image is {len(image_data) // (1024 * 1024)} MB. Venice accepts "
                f"images below 25 MB - scale the image down and try again."
            )
            return False
        
        try:
            from PIL import Image as PILImage
            width, height = PILImage.open(BytesIO(image_data)).size
        except Exception:
            # Pillow is optional here; let the API do the validating
            return True
        
        pixels = width * height
        
        if pixels < self.VENICE_MIN_IMAGE_PIXELS:
            inkex.errormsg(
                f"Image is {width}x{height}. Venice needs at least "
                f"{self.VENICE_MIN_IMAGE_PIXELS} pixels (256x256)."
            )
            return False
        
        if pixels > self.VENICE_MAX_IMAGE_PIXELS:
            inkex.errormsg(
                f"Image is {width}x{height}. Venice accepts at most "
                f"{self.VENICE_MAX_IMAGE_PIXELS} pixels."
            )
            return False
        
        return True
    
    def get_image_dimensions(self, image_data):
        """Get an image's pixel size, preferring the PNG header over Pillow."""
        size = self.get_png_size(image_data)
        if size:
            return size
        
        try:
            from PIL import Image as PILImage
            return PILImage.open(BytesIO(image_data)).size
        except Exception:
            return None
    
    def fit_to_original(self, original_data, edited_data):
        """Scale a Venice result back to the dimensions of the image it replaces.
        
        Venice answers at its own resolution tier and inferred aspect ratio, so the
        result can differ from the source. The SVG element keeps its geometry, so a
        mismatch would show up as a stretched image.
        """
        original_size = self.get_image_dimensions(original_data)
        edited_size = self.get_image_dimensions(edited_data)
        
        if not original_size or not edited_size or original_size == edited_size:
            return edited_data
        
        try:
            from PIL import Image as PILImage
        except ImportError:
            inkex.errormsg(
                f"Note: Venice returned {edited_size[0]}x{edited_size[1]} for a "
                f"{original_size[0]}x{original_size[1]} image, so it will be stretched "
                f"to fit. Install Pillow (pip install Pillow) to rescale it properly."
            )
            return edited_data
        
        try:
            image = PILImage.open(BytesIO(edited_data)).convert('RGBA')
            image = image.resize(original_size, PILImage.Resampling.LANCZOS)
            
            output = BytesIO()
            image.save(output, format='PNG')
            return output.getvalue()
        except Exception as e:
            inkex.errormsg(f"Error rescaling Venice edit: {str(e)}")
            return edited_data
    
    def apply_venice_mask(self, original_data, edited_data):
        """Composite a Venice edit back over the original through the local mask.
        
        Venice regenerates the whole frame, so this keeps the original everywhere the
        mask is opaque and the edit everywhere it is clear. This is local compositing,
        not server side inpainting - the edited region is a fresh generation.
        """
        shapes = self.get_selected_shapes_as_mask()
        
        # Nothing to composite when the whole frame was meant to be replaced
        if not shapes and self.options.mask_mode == 'full':
            return self.fit_to_original(original_data, edited_data)
        
        try:
            from PIL import Image as PILImage
        except ImportError:
            inkex.errormsg(
                "Venice has no mask channel, so partial masks are applied locally "
                "with Pillow, which is not installed.\n\n"
                "Install it with: pip install Pillow\n"
                "Or set Mask region to 'Full image (regenerate all)'."
            )
            return None
        
        try:
            original = PILImage.open(BytesIO(original_data)).convert('RGBA')
            edited = PILImage.open(BytesIO(edited_data)).convert('RGBA')
            
            # Venice may answer at a different resolution tier than the input
            if edited.size != original.size:
                edited = edited.resize(original.size, PILImage.Resampling.LANCZOS)
            
            if shapes:
                mask_data = self.create_mask_from_shapes(original_data, shapes)
            else:
                mask_data = self.create_mask(original_data, self.options.mask_mode)
            
            if not mask_data:
                return None
            
            mask = PILImage.open(BytesIO(mask_data)).convert('RGBA')
            if mask.size != original.size:
                mask = mask.resize(original.size, PILImage.Resampling.LANCZOS)
            
            # Alpha 0 marks the region to regenerate, so the alpha band selects
            # the original where it is opaque and the edit where it is clear
            keep_original = mask.split()[3]
            composite = PILImage.composite(original, edited, keep_original)
            
            output = BytesIO()
            composite.save(output, format='PNG')
            return output.getvalue()
        
        except Exception as e:
            inkex.errormsg(f"Error compositing Venice edit: {str(e)}")
            return None
    
    # ==================== Image-to-Image ====================
    
    def img2img(self, selected_image):
        """Transform image using img2img."""
        image_data = self.get_image_data(selected_image['href'])
        if not image_data:
            inkex.errormsg("Could not load image data.")
            return None
        
        # Venice infers the aspect ratio from the input image - see edit_image()
        if self.options.provider == 'venice':
            return self.img2img_venice(image_data)
        
        image_data = self.convert_image_to_rgba(image_data)
        if not image_data:
            return None
        
        if self.options.provider == 'stability':
            return self.img2img_stability(image_data)
        elif self.options.provider == 'local':
            return self.img2img_local(image_data)
        else:
            inkex.errormsg(f"img2img not supported for provider: {self.options.provider}")
            return None
    
    def img2img_stability(self, image_data):
        """Image-to-image using Stability AI."""
        engine = self.options.model or 'stable-diffusion-xl-1024-v1-0'
        url = self.PROVIDERS['stability']['img2img_url'].format(engine=engine)
        
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Accept': 'application/json'
        }
        
        boundary = '----WebKitFormBoundary' + os.urandom(16).hex()
        headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
        
        body_parts = []
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="init_image"; filename="image.png"')
        body_parts.append(b'Content-Type: image/png')
        body_parts.append(b'')
        body_parts.append(image_data)
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="init_image_mode"')
        body_parts.append(b'')
        body_parts.append(b'IMAGE_STRENGTH')
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="image_strength"')
        body_parts.append(b'')
        body_parts.append(str(1 - self.options.img2img_strength).encode())
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="text_prompts[0][text]"')
        body_parts.append(b'')
        body_parts.append(self.options.prompt.encode())
        
        if self.options.negative_prompt:
            body_parts.append(f'--{boundary}'.encode())
            body_parts.append(b'Content-Disposition: form-data; name="text_prompts[1][text]"')
            body_parts.append(b'')
            body_parts.append(self.options.negative_prompt.encode())
            
            body_parts.append(f'--{boundary}'.encode())
            body_parts.append(b'Content-Disposition: form-data; name="text_prompts[1][weight]"')
            body_parts.append(b'')
            body_parts.append(b'-1.0')
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="cfg_scale"')
        body_parts.append(b'')
        body_parts.append(str(self.options.cfg_scale).encode())
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="steps"')
        body_parts.append(b'')
        body_parts.append(str(self.options.steps).encode())
        
        if self.options.seed != -1:
            body_parts.append(f'--{boundary}'.encode())
            body_parts.append(b'Content-Disposition: form-data; name="seed"')
            body_parts.append(b'')
            body_parts.append(str(self.options.seed).encode())
        
        body_parts.append(f'--{boundary}--'.encode())
        body_parts.append(b'')
        
        body = b'\r\n'.join(body_parts)
        
        result = self.call_api_multipart(url, headers, body)
        if result and 'artifacts' in result and len(result['artifacts']) > 0:
            return base64.b64decode(result['artifacts'][0]['base64'])
        return None
    
    def img2img_local(self, image_data):
        """Image-to-image using local API."""
        endpoint = self.options.api_endpoint or 'http://127.0.0.1:7860/sdapi/v1/img2img'
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        width, height = map(int, self.get_image_size().split('x'))
        
        data = {
            'init_images': [base64.b64encode(image_data).decode('utf-8')],
            'prompt': self.options.prompt,
            'negative_prompt': self.options.negative_prompt or '',
            'denoising_strength': self.options.img2img_strength,
            'width': width,
            'height': height,
            'steps': self.options.steps,
            'cfg_scale': self.options.cfg_scale
        }
        
        if self.options.seed != -1:
            data['seed'] = self.options.seed
        
        result = self.call_api(endpoint, headers, data, use_ssl=False)
        if result and 'images' in result and len(result['images']) > 0:
            return base64.b64decode(result['images'][0])
        return None
    
    # ==================== Variations ====================
    
    def create_variation(self, selected_image):
        """Create variation of existing image."""
        if self.options.provider == 'venice':
            inkex.errormsg(
                "Venice does not provide an image variation endpoint.\n\n"
                "Use the 'Image-to-image transformation' mode instead, with a prompt "
                "describing the variation you want."
            )
            return None
        
        image_data = self.get_image_data(selected_image['href'])
        if not image_data:
            inkex.errormsg("Could not load image data for variation.")
            return None
        
        image_data = self.convert_image_to_rgba(image_data)
        if not image_data:
            return None
        
        if self.options.provider == 'openai':
            return self.variation_openai(image_data)
        else:
            # For other providers, use img2img with low strength
            self.options.img2img_strength = 0.3
            self.options.prompt = "same image with slight variations"
            
            if self.options.provider == 'stability':
                return self.img2img_stability(image_data)
            elif self.options.provider == 'local':
                return self.img2img_local(image_data)
        
        inkex.errormsg(f"Variation not supported for provider: {self.options.provider}")
        return None
    
    def variation_openai(self, image_data):
        """Create variation using OpenAI API."""
        url = self.PROVIDERS['openai']['variation_url']
        
        headers = {
            'Authorization': f'Bearer {self._api_key}'
        }
        
        boundary = '----WebKitFormBoundary' + os.urandom(16).hex()
        headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
        
        body_parts = []
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="image"; filename="image.png"')
        body_parts.append(b'Content-Type: image/png')
        body_parts.append(b'')
        body_parts.append(image_data)
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="model"')
        body_parts.append(b'')
        body_parts.append(b'dall-e-2')
        
        size = self.get_image_size()
        if size not in ['256x256', '512x512', '1024x1024']:
            size = '1024x1024'
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="size"')
        body_parts.append(b'')
        body_parts.append(size.encode())
        
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Disposition: form-data; name="response_format"')
        body_parts.append(b'')
        body_parts.append(b'b64_json')
        
        body_parts.append(f'--{boundary}--'.encode())
        body_parts.append(b'')
        
        body = b'\r\n'.join(body_parts)
        
        result = self.call_api_multipart(url, headers, body)
        if result and 'data' in result and len(result['data']) > 0:
            if 'b64_json' in result['data'][0]:
                return base64.b64decode(result['data'][0]['b64_json'])
            elif 'url' in result['data'][0]:
                return self.download_image(result['data'][0]['url'])
        return None
    
    # ==================== API Calls ====================
    
    def get_ssl_context(self):
        """Get proper SSL context."""
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            return context
        except:
            return ssl.create_default_context()
    
    def get_url_opener(self, use_ssl=True):
        """Build a URL opener with the configured proxy and SSL context."""
        context = self.get_ssl_context() if use_ssl else None
        
        # Setup proxy if configured
        if self.options.use_proxy and self.options.proxy_url:
            proxy_handler = urllib.request.ProxyHandler({
                'http': self.options.proxy_url,
                'https': self.options.proxy_url
            })
            if context:
                https_handler = urllib.request.HTTPSHandler(context=context)
                return urllib.request.build_opener(proxy_handler, https_handler)
            return urllib.request.build_opener(proxy_handler)
        
        if context:
            https_handler = urllib.request.HTTPSHandler(context=context)
            return urllib.request.build_opener(https_handler)
        
        return urllib.request.build_opener()
    
    def extract_api_error(self, error_body, fallback):
        """Get a readable message out of a provider error response.
        
        Handles the OpenAI shape ({"error": {"message": ...}}) and the Venice shape
        ({"error": "...", "details": {...}}), falling back to the raw error.
        """
        try:
            if isinstance(error_body, bytes):
                error_body = error_body.decode('utf-8', 'replace')
            error_data = json.loads(error_body)
        except Exception:
            return fallback
        
        if not isinstance(error_data, dict):
            return fallback
        
        error = error_data.get('error')
        
        if isinstance(error, dict):
            return error.get('message', fallback)
        
        if isinstance(error, str) and error:
            details = error_data.get('details')
            if details:
                return f"{error} ({json.dumps(details)})"
            return error
        
        if isinstance(error_data.get('message'), str):
            return error_data['message']
        
        return fallback
    
    def call_api(self, url, headers, data, use_ssl=True):
        """Call API with JSON data and retry logic."""
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        opener = self.get_url_opener(use_ssl)
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with opener.open(req, timeout=180) as response:
                    return json.loads(response.read().decode('utf-8'))
            
            except urllib.error.HTTPError as e:
                error_message = self.extract_api_error(e.read(), str(e))
                
                # Check if retryable
                if e.code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    continue
                
                inkex.errormsg(f"API Error: {error_message}")
                return None
            
            except urllib.error.URLError as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                inkex.errormsg(f"Connection Error: {str(e)}")
                return None
            
            except Exception as e:
                inkex.errormsg(f"Error: {str(e)}")
                return None
        
        return None
    
    def call_api_get(self, url, headers):
        """Call API with GET request."""
        req = urllib.request.Request(url, headers=headers, method='GET')
        
        context = self.get_ssl_context()
        
        try:
            with urllib.request.urlopen(req, timeout=60, context=context) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            return None
    
    def call_api_multipart(self, url, headers, body):
        """Call API with multipart form data and retry logic."""
        req = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method='POST'
        )
        
        context = self.get_ssl_context()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=180, context=context) as response:
                    return json.loads(response.read().decode('utf-8'))
            
            except urllib.error.HTTPError as e:
                error_message = self.extract_api_error(e.read(), str(e))
                
                if e.code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    continue
                
                inkex.errormsg(f"API Error: {error_message}")
                return None
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                inkex.errormsg(f"Error: {str(e)}")
                return None
        
        return None
    
    def call_api_binary(self, url, headers, data):
        """Call API with JSON data and return raw image bytes.
        
        Venice's edit endpoints answer with image/png rather than JSON.
        """
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        opener = self.get_url_opener()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with opener.open(req, timeout=180) as response:
                    content_type = response.headers.get('Content-Type', '')
                    body = response.read()
                    
                    # An image was expected, so a JSON body carries an error
                    if 'application/json' in content_type:
                        inkex.errormsg(
                            f"API Error: "
                            f"{self.extract_api_error(body, 'unexpected JSON response')}"
                        )
                        return None
                    
                    if response.headers.get('x-venice-is-content-violation') == 'true':
                        inkex.errormsg(
                            "Venice rejected this request under its content policy."
                        )
                        return None
                    
                    warning = response.headers.get('x-venice-model-deprecation-warning')
                    if warning:
                        inkex.errormsg(f"Venice model warning: {warning}")
                    
                    return body
            
            except urllib.error.HTTPError as e:
                error_message = self.extract_api_error(e.read(), str(e))
                
                if e.code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    continue
                
                inkex.errormsg(f"API Error: {error_message}")
                return None
            
            except urllib.error.URLError as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                inkex.errormsg(f"Connection Error: {str(e)}")
                return None
            
            except Exception as e:
                inkex.errormsg(f"Error: {str(e)}")
                return None
        
        return None
    
    # ==================== Image Data Helpers ====================
    
    def get_image_data(self, href):
        """Get image data from href."""
        if href.startswith('data:'):
            try:
                header, encoded = href.split(',', 1)
                return base64.b64decode(encoded)
            except:
                return None
        elif href.startswith('file://') or os.path.isabs(href):
            try:
                file_path = href.replace('file://', '')
                with open(file_path, 'rb') as f:
                    return f.read()
            except:
                return None
        else:
            try:
                with open(href, 'rb') as f:
                    return f.read()
            except:
                return None
    
    def download_image(self, image_url):
        """Download image from URL."""
        context = self.get_ssl_context()
        
        try:
            with urllib.request.urlopen(image_url, timeout=60, context=context) as response:
                return response.read()
        except Exception as e:
            inkex.errormsg(f"Error downloading image: {str(e)}")
            return None
    
    def save_image_to_disk(self, image_data):
        """Save image data to disk."""
        if not self.options.save_to_disk:
            return None
        
        save_dir = self.get_save_directory()
        
        try:
            os.makedirs(save_dir, exist_ok=True)
        except:
            inkex.errormsg(f"Could not create directory: {save_dir}")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        seed_str = f"_seed{self.options.seed}" if self.options.seed != -1 else ""
        filename = f"{self.options.filename_prefix}_{timestamp}{seed_str}.png"
        filepath = os.path.join(save_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                f.write(image_data)
            return filepath
        except Exception as e:
            inkex.errormsg(f"Error saving image: {str(e)}")
            return None
    
    # ==================== Document Manipulation ====================
    
    def add_image_to_document(self, image_data, offset=0):
        """Add image to document."""
        if isinstance(image_data, str):
            image_data = self.download_image(image_data)
            if not image_data:
                return
        
        saved_path = self.save_image_to_disk(image_data)
        
        image_elem = Image()
        image_elem.set('id', self.svg.get_unique_id('ai-image'))
        
        if self.options.embed_in_svg:
            encoded = base64.b64encode(image_data).decode('utf-8')
            image_elem.set('xlink:href', f'data:image/png;base64,{encoded}')
        elif saved_path:
            image_elem.set('xlink:href', saved_path)
        
        position = self.calculate_position()
        size = self.calculate_size()
        
        image_elem.set('x', str(position['x'] + offset))
        image_elem.set('y', str(position['y'] + offset))
        image_elem.set('width', str(size['width']))
        image_elem.set('height', str(size['height']))
        
        self.svg.get_current_layer().append(image_elem)
    
    def replace_image(self, image_elem, image_data):
        """Replace existing image with new one."""
        if isinstance(image_data, str):
            image_data = self.download_image(image_data)
            if not image_data:
                return
        
        saved_path = self.save_image_to_disk(image_data)
        
        if self.options.embed_in_svg:
            encoded = base64.b64encode(image_data).decode('utf-8')
            image_elem.set('xlink:href', f'data:image/png;base64,{encoded}')
        elif saved_path:
            image_elem.set('xlink:href', saved_path)
    
    def calculate_position(self):
        """Calculate position based on position mode."""
        doc_width = self.svg.viewport_width
        doc_height = self.svg.viewport_height
        
        size = self.calculate_size()
        
        positions = {
            'center': {
                'x': (doc_width - size['width']) / 2,
                'y': (doc_height - size['height']) / 2
            },
            'top_left': {'x': 0, 'y': 0},
            'top_center': {
                'x': (doc_width - size['width']) / 2,
                'y': 0
            },
            'top_right': {
                'x': doc_width - size['width'],
                'y': 0
            },
            'bottom_left': {
                'x': 0,
                'y': doc_height - size['height']
            },
            'bottom_center': {
                'x': (doc_width - size['width']) / 2,
                'y': doc_height - size['height']
            },
            'bottom_right': {
                'x': doc_width - size['width'],
                'y': doc_height - size['height']
            },
            'cursor': {
                'x': (doc_width - size['width']) / 2,
                'y': (doc_height - size['height']) / 2
            }
        }
        
        if self.options.position_mode == 'cursor' and self.svg.selection:
            for elem in self.svg.selection:
                bbox = elem.bounding_box()
                if bbox:
                    return {
                        'x': bbox.center_x - size['width']/2,
                        'y': bbox.center_y - size['height']/2
                    }
        
        return positions.get(self.options.position_mode, positions['center'])
    
    def calculate_size(self):
        """Calculate image size based on scale mode."""
        size_str = self.get_image_size()
        width, height = map(int, size_str.split('x'))
        
        doc_width = self.svg.viewport_width
        doc_height = self.svg.viewport_height
        
        if self.options.scale_mode == 'original':
            return {'width': width, 'height': height}
        
        elif self.options.scale_mode == 'fit_width':
            scale = doc_width / width
            return {'width': doc_width, 'height': height * scale}
        
        elif self.options.scale_mode == 'fit_height':
            scale = doc_height / height
            return {'width': width * scale, 'height': doc_height}
        
        elif self.options.scale_mode == 'fit_canvas':
            scale_w = doc_width / width
            scale_h = doc_height / height
            scale = min(scale_w, scale_h)
            return {'width': width * scale, 'height': height * scale}
        
        elif self.options.scale_mode == 'custom':
            return {
                'width': self.options.placement_width,
                'height': self.options.placement_height
            }
        
        return {'width': width, 'height': height}


if __name__ == '__main__':
    AIImageGenerator().run()