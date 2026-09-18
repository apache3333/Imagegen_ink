#!/usr/bin/env python3
"""
Tests for the Venice AI provider in img_gen.py.

Uses only the standard library. Inkscape's `inkex` module is stubbed, and the HTTP
calls are replaced with recorded fakes, so no network access and no API key are needed.

Run from the extension directory:

    python3 test_venice.py
"""

import base64
import json
import os
import re
import sys
import types
import unittest
from types import SimpleNamespace


# ==================== inkex stub ====================

ERRORS = []


def _build_inkex_stub():
    """Build the minimal inkex surface img_gen.py imports at module level."""
    inkex = types.ModuleType('inkex')

    class EffectExtension:
        def __init__(self):
            self.options = SimpleNamespace()

        def run(self):
            raise NotImplementedError

    class Image:
        pass

    class Group:
        pass

    inkex.EffectExtension = EffectExtension
    inkex.Image = Image
    inkex.Group = Group
    inkex.Boolean = bool
    inkex.errormsg = ERRORS.append
    return inkex


sys.modules.setdefault('inkex', _build_inkex_stub())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from img_gen import AIImageGenerator  # noqa: E402


try:
    from PIL import Image as PILImage  # noqa: F401
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ==================== Helpers ====================

def png_bytes(width, height):
    """Build a byte string with a valid PNG signature and IHDR dimensions."""
    return (b'\x89PNG\r\n\x1a\n'
            + (13).to_bytes(4, 'big') + b'IHDR'
            + width.to_bytes(4, 'big') + height.to_bytes(4, 'big')
            + b'\x08\x06\x00\x00\x00'
            + b'\x00' * 16)


def make_extension(**overrides):
    """Build an extension instance with predictable options and no config file."""
    ext = AIImageGenerator()
    ext._config = {}
    ext._api_key = 'test-key-not-a-real-credential'

    options = {
        'provider': 'venice',
        'model': 'venice-sd35',
        'prompt': 'a lighthouse in fog',
        'negative_prompt': '',
        'edit_instruction': 'add a rainbow',
        'image_size': '1024x1024',
        'use_custom_size': False,
        'custom_width': 1024,
        'custom_height': 1024,
        'cfg_scale': 7.0,
        'steps': 30,
        'seed': -1,
        'img2img_strength': 0.75,
        'mask_mode': 'full',
        'use_selection_as_mask': False,
        'mask_feather': 0,
        'hide_watermark': True,
        'use_proxy': False,
        'proxy_url': '',
    }
    options.update(overrides)
    ext.options = SimpleNamespace(**options)
    return ext


def record_generate(ext, images=None):
    """Replace call_api and record the request it would have sent."""
    sent = {}

    def fake_call_api(url, headers, data, use_ssl=True):
        sent['url'] = url
        sent['headers'] = headers
        sent['data'] = data
        if images is None:
            return {'images': [base64.b64encode(png_bytes(1024, 1024)).decode()]}
        return {'images': images}

    ext.call_api = fake_call_api
    return sent


def record_edit(ext, response=None):
    """Replace call_api_binary and record the request it would have sent."""
    sent = {}

    def fake_call_api_binary(url, headers, data):
        sent['url'] = url
        sent['headers'] = headers
        sent['data'] = data
        return png_bytes(1024, 1024) if response is None else response

    ext.call_api_binary = fake_call_api_binary
    return sent


class VeniceTestCase(unittest.TestCase):
    def setUp(self):
        del ERRORS[:]


# ==================== Generation ====================

class TestVeniceGenerate(VeniceTestCase):

    def test_pixel_model_sends_width_and_height(self):
        ext = make_extension(model='venice-sd35', image_size='1024x1024')
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['url'], 'https://api.venice.ai/api/v1/image/generate')
        self.assertEqual(sent['data']['model'], 'venice-sd35')
        self.assertEqual(sent['data']['width'], 1024)
        self.assertEqual(sent['data']['height'], 1024)
        self.assertNotIn('aspect_ratio', sent['data'])

    def test_aspect_model_sends_aspect_ratio_only(self):
        ext = make_extension(model='qwen-image-3', image_size='1792x1024')
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['data']['aspect_ratio'], '16:9')
        self.assertNotIn('width', sent['data'])
        self.assertNotIn('height', sent['data'])

    def test_png_format_is_requested(self):
        """The extension embeds and saves PNG; Venice would default to webp."""
        ext = make_extension()
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['data']['format'], 'png')

    def test_hide_watermark_is_sent_on_generation(self):
        ext = make_extension(hide_watermark=True)
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertIs(sent['data']['hide_watermark'], True)

    def test_hide_watermark_can_be_turned_off(self):
        ext = make_extension(hide_watermark=False)
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertIs(sent['data']['hide_watermark'], False)

    def test_bearer_auth_header(self):
        ext = make_extension()
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['headers']['Authorization'],
                         'Bearer test-key-not-a-real-credential')

    def test_oversized_request_is_capped_and_aligned(self):
        """1792x1024 must fit 1280px and land on the model's 16px step."""
        ext = make_extension(model='venice-sd35', image_size='1792x1024')
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['data']['width'], 1280)
        self.assertEqual(sent['data']['height'], 736)
        self.assertEqual(sent['data']['width'] % 16, 0)
        self.assertEqual(sent['data']['height'] % 16, 0)

    def test_aspect_ratio_mapping(self):
        cases = {
            '1024x1024': '1:1',
            '1792x1024': '16:9',
            '1024x1792': '9:16',
            '512x512': '1:1',
        }
        for size, expected in cases.items():
            with self.subTest(size=size):
                ext = make_extension(model='qwen-image-3', image_size=size)
                self.assertEqual(ext.get_venice_aspect_ratio(), expected)

    def test_steps_clamped_to_model_maximum(self):
        ext = make_extension(model='z-image-turbo', steps=30)
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['data']['steps'], 8)
        self.assertTrue(any('sampling steps' in e for e in ERRORS))

    def test_steps_within_limit_are_untouched(self):
        ext = make_extension(model='venice-sd35', steps=25)
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['data']['steps'], 25)
        self.assertEqual(ERRORS, [])

    def test_random_seed_is_omitted(self):
        ext = make_extension(seed=-1)
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertNotIn('seed', sent['data'])

    def test_seed_passed_through(self):
        ext = make_extension(seed=42)
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['data']['seed'], 42)

    def test_out_of_range_seed_is_clamped_with_notice(self):
        """The dialog allows seeds up to 2147483647; Venice stops at 999999999."""
        ext = make_extension(seed=2147483647)
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['data']['seed'], 999999999)
        self.assertTrue(any('clamped' in e for e in ERRORS))

    def test_negative_prompt_uses_the_real_field(self):
        ext = make_extension(negative_prompt='blurry, watermark')
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['data']['negative_prompt'], 'blurry, watermark')
        self.assertNotIn('Avoid:', sent['data']['prompt'])

    def test_overlong_prompt_is_rejected_before_calling_the_api(self):
        ext = make_extension(model='venice-sd35', prompt='x' * 1501)
        sent = record_generate(ext)

        self.assertIsNone(ext.generate_venice())
        self.assertEqual(sent, {})
        self.assertTrue(any('1500' in e for e in ERRORS))

    def test_non_venice_model_falls_back_with_notice(self):
        ext = make_extension(model='dall-e-3')
        sent = record_generate(ext)
        ext.generate_venice()

        self.assertEqual(sent['data']['model'], 'venice-sd35')
        self.assertTrue(any('dall-e-3' in e for e in ERRORS))

    def test_result_is_decoded_from_base64(self):
        expected = png_bytes(640, 480)
        ext = make_extension()
        record_generate(ext, images=[base64.b64encode(expected).decode()])

        self.assertEqual(ext.generate_venice(), expected)

    def test_placement_size_follows_the_returned_image(self):
        """An aspect-ratio model can answer at other dimensions than requested."""
        ext = make_extension(model='qwen-image-3', image_size='1024x1024')
        record_generate(ext, images=[base64.b64encode(png_bytes(1536, 1024)).decode()])
        ext.generate_venice()

        self.assertTrue(ext.options.use_custom_size)
        self.assertEqual(ext.options.custom_width, 1536)
        self.assertEqual(ext.options.custom_height, 1024)


# ==================== Editing ====================

class TestVeniceEdit(VeniceTestCase):

    def test_edit_request_shape(self):
        image = png_bytes(1024, 1024)
        ext = make_extension(model='firered-image-edit', edit_instruction='add a rainbow')
        sent = record_edit(ext)
        ext.edit_venice(image)

        self.assertEqual(sent['url'], 'https://api.venice.ai/api/v1/image/edit')
        self.assertEqual(sent['data']['prompt'], 'add a rainbow')
        self.assertEqual(sent['data']['output_format'], 'png')
        self.assertEqual(base64.b64decode(sent['data']['image']), image)

    def test_edit_uses_model_not_modelid(self):
        """/image/edit takes `model`; `modelId` is deprecated there."""
        ext = make_extension(model='firered-image-edit')
        sent = record_edit(ext)
        ext.edit_venice(png_bytes(1024, 1024))

        self.assertIn('model', sent['data'])
        self.assertNotIn('modelId', sent['data'])

    def test_hide_watermark_is_not_sent_on_edits(self):
        """/image/edit has no hide_watermark field; sending it would be a 400."""
        ext = make_extension(model='firered-image-edit', hide_watermark=True)
        sent = record_edit(ext)
        ext.edit_venice(png_bytes(1024, 1024))

        self.assertNotIn('hide_watermark', sent['data'])

    def test_aspect_ratio_is_left_to_the_api(self):
        """Omitting aspect_ratio makes Venice infer it from the input image."""
        ext = make_extension()
        sent = record_edit(ext)
        ext.edit_venice(png_bytes(1024, 1024))

        self.assertNotIn('aspect_ratio', sent['data'])

    def test_generation_model_falls_back_to_an_edit_model(self):
        ext = make_extension(model='qwen-image-3')
        sent = record_edit(ext)
        ext.edit_venice(png_bytes(1024, 1024))

        self.assertEqual(sent['data']['model'], 'firered-image-edit')
        self.assertTrue(any('cannot edit images' in e for e in ERRORS))

    def test_full_mask_returns_the_api_result_unchanged(self):
        """Matching dimensions need no rescale."""
        expected = png_bytes(1024, 1024)
        ext = make_extension(mask_mode='full')
        record_edit(ext, response=expected)

        self.assertEqual(ext.edit_venice(png_bytes(1024, 1024)), expected)

    def test_mismatched_result_is_flagged_without_pillow(self):
        """Venice answers at its own resolution tier, which would stretch the image."""
        ext = make_extension(mask_mode='full')
        record_edit(ext, response=png_bytes(1536, 1024))
        result = ext.edit_venice(png_bytes(1024, 1024))

        self.assertIsNotNone(result)
        if not HAS_PIL:
            self.assertTrue(any('stretched' in e for e in ERRORS))

    @unittest.skipUnless(HAS_PIL, 'Pillow is not installed')
    def test_mismatched_result_is_rescaled_with_pillow(self):
        from io import BytesIO
        from PIL import Image as PIL

        def solid(size):
            buffer = BytesIO()
            PIL.new('RGBA', size, (0, 0, 255, 255)).save(buffer, format='PNG')
            return buffer.getvalue()

        ext = make_extension(mask_mode='full')
        record_edit(ext, response=solid((1536, 1024)))
        result = ext.edit_venice(solid((1024, 1024)))

        self.assertEqual(PIL.open(BytesIO(result)).size, (1024, 1024))

    def test_negative_prompt_is_folded_into_the_edit_instruction(self):
        """The edit endpoint has no negative_prompt field."""
        ext = make_extension(edit_instruction='add a rainbow',
                             negative_prompt='cartoon')
        sent = record_edit(ext)
        ext.edit_venice(png_bytes(1024, 1024))

        self.assertEqual(sent['data']['prompt'], 'add a rainbow. Avoid: cartoon')
        self.assertNotIn('negative_prompt', sent['data'])

    def test_region_hint_is_added_for_a_partial_mask(self):
        """Venice is not told where the mask is, so the region is named in words."""
        ext = make_extension(model='firered-image-edit',
                             edit_instruction='add a dwarf sitting',
                             mask_mode='center')
        sent = record_edit(ext)
        ext.edit_venice(png_bytes(1024, 1024))

        self.assertEqual(sent['data']['prompt'],
                         'add a dwarf sitting, in the centre of the image')

    def test_region_hint_matches_the_mask_mode(self):
        for mask_mode, expected in AIImageGenerator.VENICE_MASK_HINTS.items():
            with self.subTest(mask_mode=mask_mode):
                ext = make_extension(model='firered-image-edit',
                                     edit_instruction='add a bird',
                                     mask_mode=mask_mode)
                sent = record_edit(ext)
                ext.edit_venice(png_bytes(1024, 1024))

                self.assertEqual(sent['data']['prompt'], f'add a bird, {expected}')

    def test_no_region_hint_for_a_full_frame_edit(self):
        ext = make_extension(model='firered-image-edit',
                             edit_instruction='make it snowy',
                             mask_mode='full')
        sent = record_edit(ext)
        ext.edit_venice(png_bytes(1024, 1024))

        self.assertEqual(sent['data']['prompt'], 'make it snowy')

    def test_region_hint_does_not_double_the_punctuation(self):
        ext = make_extension(model='firered-image-edit',
                             edit_instruction='add a dwarf sitting.',
                             mask_mode='center')
        sent = record_edit(ext)
        ext.edit_venice(png_bytes(1024, 1024))

        self.assertNotIn('.,', sent['data']['prompt'])

    def test_region_hint_combines_with_a_negative_prompt(self):
        ext = make_extension(model='firered-image-edit',
                             edit_instruction='add a dwarf sitting',
                             negative_prompt='cartoon',
                             mask_mode='center')
        sent = record_edit(ext)
        ext.edit_venice(png_bytes(1024, 1024))

        self.assertEqual(sent['data']['prompt'],
                         'add a dwarf sitting, in the centre of the image. Avoid: cartoon')

    def test_shape_region_is_described_by_grid_position(self):
        """Selected shapes get a position rather than a mask mode."""
        ext = make_extension(model='firered-image-edit',
                             edit_instruction='add a lamp',
                             use_selection_as_mask=True)
        ext.svg = SimpleNamespace(viewport_width=1000, viewport_height=1000,
                                  selection=[])
        shape = SimpleNamespace(
            bounding_box=lambda: SimpleNamespace(center_x=100, center_y=900))
        ext.get_selected_shapes_as_mask = lambda: [shape]

        self.assertEqual(ext.add_venice_region_hint('add a lamp'),
                         'add a lamp, in the bottom left area of the image')

    def test_shape_region_falls_back_quietly(self):
        ext = make_extension(model='firered-image-edit', use_selection_as_mask=True)
        ext.get_selected_shapes_as_mask = lambda: [SimpleNamespace()]

        self.assertEqual(ext.add_venice_region_hint('add a lamp'), 'add a lamp')

    def test_missing_instruction_is_rejected(self):
        ext = make_extension(edit_instruction='')
        sent = record_edit(ext)

        self.assertIsNone(ext.edit_venice(png_bytes(1024, 1024)))
        self.assertEqual(sent, {})

    def test_oversized_image_is_rejected_before_upload(self):
        ext = make_extension()
        sent = record_edit(ext)
        huge = png_bytes(1024, 1024) + b'\x00' * (25 * 1024 * 1024)

        self.assertIsNone(ext.edit_venice(huge))
        self.assertEqual(sent, {})
        self.assertTrue(any('25 MB' in e for e in ERRORS))

    @unittest.skipUnless(HAS_PIL, 'Pillow is not installed')
    def test_partial_mask_keeps_the_original_outside_the_mask(self):
        from io import BytesIO
        from PIL import Image as PIL

        def solid(color):
            buffer = BytesIO()
            PIL.new('RGBA', (64, 64), color).save(buffer, format='PNG')
            return buffer.getvalue()

        original = solid((255, 0, 0, 255))
        edited = solid((0, 0, 255, 255))

        ext = make_extension(mask_mode='top_half')
        result = ext.apply_venice_mask(original, edited)
        self.assertIsNotNone(result)

        composite = PIL.open(BytesIO(result)).convert('RGBA')
        # top_half clears the alpha over the top, so the edit lands there
        self.assertEqual(composite.getpixel((32, 8))[:3], (0, 0, 255))
        self.assertEqual(composite.getpixel((32, 56))[:3], (255, 0, 0))

    @unittest.skipUnless(HAS_PIL, 'Pillow is not installed')
    def test_edit_is_resized_to_the_original_before_compositing(self):
        from io import BytesIO
        from PIL import Image as PIL

        def solid(size, color):
            buffer = BytesIO()
            PIL.new('RGBA', size, color).save(buffer, format='PNG')
            return buffer.getvalue()

        original = solid((64, 64), (255, 0, 0, 255))
        edited = solid((128, 128), (0, 0, 255, 255))

        ext = make_extension(mask_mode='top_half')
        result = ext.apply_venice_mask(original, edited)

        self.assertEqual(PIL.open(BytesIO(result)).size, (64, 64))


# ==================== Img2Img and variation ====================

class TestVeniceOtherModes(VeniceTestCase):

    def test_img2img_uses_the_edit_endpoint_with_the_prompt(self):
        ext = make_extension(prompt='oil painting')
        sent = record_edit(ext)
        ext.img2img_venice(png_bytes(1024, 1024))

        self.assertEqual(sent['url'], 'https://api.venice.ai/api/v1/image/edit')
        self.assertEqual(sent['data']['prompt'], 'oil painting')

    def test_img2img_syncs_the_placement_box(self):
        """img2img places a new image, so the box must match the result."""
        ext = make_extension(model='firered-image-edit', image_size='1024x1024')
        record_edit(ext, response=png_bytes(1536, 1024))
        ext.img2img_venice(png_bytes(1024, 1024))

        self.assertTrue(ext.options.use_custom_size)
        self.assertEqual(ext.options.custom_width, 1536)
        self.assertEqual(ext.options.custom_height, 1024)

    def test_changed_strength_is_reported_as_ignored(self):
        ext = make_extension(model='firered-image-edit', img2img_strength=0.3)
        record_edit(ext)
        ext.img2img_venice(png_bytes(1024, 1024))

        self.assertTrue(any('strength' in e for e in ERRORS))

    def test_default_strength_does_not_warn(self):
        ext = make_extension(model='firered-image-edit', img2img_strength=0.75)
        record_edit(ext)
        ext.img2img_venice(png_bytes(1024, 1024))

        self.assertEqual(ERRORS, [])

    def test_variation_is_rejected_with_a_pointer_to_img2img(self):
        ext = make_extension()
        result = ext.create_variation({'href': 'data:image/png;base64,', 'element': None})

        self.assertIsNone(result)
        self.assertTrue(any('Image-to-image' in e for e in ERRORS))


# ==================== Shared plumbing ====================

class TestErrorExtraction(VeniceTestCase):
    """extract_api_error is shared, so the existing shapes must keep working."""

    def setUp(self):
        super().setUp()
        self.ext = make_extension()

    def test_openai_shape_is_unchanged(self):
        body = json.dumps({'error': {'message': 'Invalid API key'}})
        self.assertEqual(self.ext.extract_api_error(body, 'fallback'), 'Invalid API key')

    def test_venice_string_shape(self):
        body = json.dumps({'error': 'Insufficient balance'})
        self.assertEqual(self.ext.extract_api_error(body, 'fallback'),
                         'Insufficient balance')

    def test_venice_detailed_shape_includes_details(self):
        body = json.dumps({'error': 'Invalid request',
                           'details': {'width': {'_errors': ['Too large']}}})
        message = self.ext.extract_api_error(body, 'fallback')

        self.assertIn('Invalid request', message)
        self.assertIn('Too large', message)

    def test_bytes_are_accepted(self):
        body = json.dumps({'error': 'Rate limit exceeded'}).encode('utf-8')
        self.assertEqual(self.ext.extract_api_error(body, 'fallback'),
                         'Rate limit exceeded')

    def test_non_json_falls_back(self):
        self.assertEqual(self.ext.extract_api_error('<html>502</html>', 'fallback'),
                         'fallback')

    def test_unknown_shape_falls_back(self):
        body = json.dumps({'error': {'code': 500}})
        self.assertEqual(self.ext.extract_api_error(body, 'fallback'), 'fallback')


class TestPngSize(VeniceTestCase):

    def test_reads_dimensions_without_pillow(self):
        ext = make_extension()
        self.assertEqual(ext.get_png_size(png_bytes(1536, 1024)), (1536, 1024))

    def test_rejects_non_png(self):
        ext = make_extension()
        self.assertIsNone(ext.get_png_size(b'\xff\xd8\xff\xe0 not a png'))

    def test_rejects_truncated_data(self):
        ext = make_extension()
        self.assertIsNone(ext.get_png_size(b'\x89PNG\r\n\x1a\n'))


class TestProviderRegistration(VeniceTestCase):

    def test_venice_entry(self):
        venice = AIImageGenerator.PROVIDERS['venice']

        self.assertEqual(venice['env_key'], 'VENICE_API_KEY')
        self.assertEqual(venice['config_key'], 'venice_api_key')
        self.assertTrue(venice['generate_url'].startswith('https://api.venice.ai/api/v1'))

    def test_existing_providers_are_untouched(self):
        providers = AIImageGenerator.PROVIDERS

        self.assertEqual(providers['openai']['env_key'], 'OPENAI_API_KEY')
        self.assertEqual(providers['stability']['env_key'], 'STABILITY_API_KEY')
        self.assertEqual(providers['replicate']['env_key'], 'REPLICATE_API_TOKEN')
        self.assertEqual(providers['local']['env_key'], '')

    def test_every_dropdown_model_is_known(self):
        """Venice models offered in img_gen.inx must be resolvable in img_gen.py."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img_gen.inx')
        with open(path, encoding='utf-8') as handle:
            inx = handle.read()

        known = set(AIImageGenerator.VENICE_MODELS) | set(AIImageGenerator.VENICE_EDIT_MODELS)
        for model in known:
            self.assertIn(f'<option value="{model}">', inx,
                          f'{model} is missing from the Model dropdown')

    def test_provider_dropdown_matches_the_providers_table(self):
        inx = self._read_inx()
        options = re.findall(r'<option value="([^"]+)">', self._param_block(inx, 'provider'))

        self.assertEqual(set(options), set(AIImageGenerator.PROVIDERS))

    def test_every_dialog_param_has_an_argument(self):
        """An .inx param with no add_argument makes the extension fail to launch."""
        inx = self._read_inx()
        params = set(re.findall(r'<param name="([^"]+)"', inx))

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img_gen.py')
        with open(path, encoding='utf-8') as handle:
            arguments = set(re.findall(r'pars\.add_argument\(\s*"--([a-z_0-9]+)"',
                                       handle.read()))

        self.assertEqual(params - arguments, set())

    @staticmethod
    def _read_inx():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img_gen.inx')
        with open(path, encoding='utf-8') as handle:
            return handle.read()

    @staticmethod
    def _param_block(inx, name):
        start = inx.index(f'<param name="{name}"')
        return inx[start:inx.index('</param>', start)]

    def test_api_key_resolution_uses_the_venice_environment_variable(self):
        ext = make_extension()
        ext.options.api_key = ''
        ext.options.use_env_key = True
        ext.options.use_config_key = False
        ext.options.save_api_key = False

        os.environ['VENICE_API_KEY'] = 'env-key-not-a-real-credential'
        try:
            self.assertEqual(ext.get_api_key(), 'env-key-not-a-real-credential')
        finally:
            del os.environ['VENICE_API_KEY']


if __name__ == '__main__':
    unittest.main(verbosity=2)
