#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contract tests for subsai.configs schemas (no models loaded)."""
import pytest

from subsai.configs import ADVANCED_TOOLS_CONFIGS, AVAILABLE_MODELS, BASIC_TOOLS_CONFIGS
from subsai.models.abstract_model import AbstractModel

SCHEMA_FIELDS = ['type', 'description', 'options', 'default']


def all_schemas():
    schemas = {}
    for name, entry in AVAILABLE_MODELS.items():
        schemas[f'model:{name}'] = entry['config_schema']
    for group in (BASIC_TOOLS_CONFIGS, ADVANCED_TOOLS_CONFIGS):
        for name, entry in group.items():
            schemas[f'tools:{name}'] = entry['config_schema']
    return schemas


SCHEMA_CASES = all_schemas()


class TestAvailableModelsRegistry:
    def test_registry_not_empty(self):
        assert len(AVAILABLE_MODELS) >= 4

    @pytest.mark.parametrize('model_name', list(AVAILABLE_MODELS))
    def test_entry_shape(self, model_name):
        entry = AVAILABLE_MODELS[model_name]
        for key in ['class', 'description', 'url', 'config_schema']:
            assert key in entry, f'{model_name} missing {key}'
        assert issubclass(entry['class'], AbstractModel)
        assert entry['description']
        assert entry['url'].startswith('http')

    @pytest.mark.parametrize('model_name', list(AVAILABLE_MODELS))
    def test_class_model_name_matches_key(self, model_name):
        assert AVAILABLE_MODELS[model_name]['class'].model_name == model_name


class TestSchemaContract:
    """The webui renders widgets purely from these schemas, so every field
    must expose the 4 documented keys and a type-consistent default."""

    @pytest.mark.parametrize('schema_name', list(SCHEMA_CASES))
    def test_every_field_has_required_keys(self, schema_name):
        schema = SCHEMA_CASES[schema_name]
        assert isinstance(schema, dict) and schema
        for config_name, field in schema.items():
            for key in SCHEMA_FIELDS:
                assert key in field, f'{schema_name}.{config_name} missing {key}'

    @pytest.mark.parametrize('schema_name', list(SCHEMA_CASES))
    def test_defaults_consistent_with_declared_type(self, schema_name):
        schema = SCHEMA_CASES[schema_name]
        for config_name, field in schema.items():
            default = field['default']
            declared = field['type']
            if default is None:
                continue
            if declared is bool:
                assert isinstance(default, bool), f'{schema_name}.{config_name}'
            elif declared is int:
                assert isinstance(default, int), f'{schema_name}.{config_name}'
            elif declared is float:
                assert isinstance(default, (int, float)), f'{schema_name}.{config_name}'
            elif declared is str:
                assert isinstance(default, str), f'{schema_name}.{config_name}'
            elif declared is list:
                assert isinstance(default, (list, str, int, float)), \
                    f'{schema_name}.{config_name}'

    @pytest.mark.parametrize('schema_name', list(SCHEMA_CASES))
    def test_list_options_contain_default(self, schema_name):
        schema = SCHEMA_CASES[schema_name]
        for config_name, field in schema.items():
            if field['type'] is list and field['options'] and field['default'] is not None:
                assert field['default'] in field['options'], \
                    f'{schema_name}.{config_name}: default {field["default"]!r} ' \
                    f'not in options'


class TestToolsConfigs:
    def test_basic_tools_entries(self):
        for name in ['set time', 'shift']:
            assert name in BASIC_TOOLS_CONFIGS
            assert 'config_schema' in BASIC_TOOLS_CONFIGS[name]

    def test_shift_has_frames_and_fps(self):
        schema = BASIC_TOOLS_CONFIGS['shift']['config_schema']
        assert 'frames' in schema and 'fps' in schema

    def test_translation_options_are_real_models(self):
        from subsai.utils import available_translation_models
        schema = ADVANCED_TOOLS_CONFIGS['Translation']['config_schema']
        assert schema['model']['options'] == available_translation_models()
        assert schema['model']['default'] == available_translation_models()[0]

    def test_device_options_start_with_auto(self):
        schema = ADVANCED_TOOLS_CONFIGS['Translation']['config_schema']
        assert schema['device']['options'][0] == 'auto'
        assert 'cpu' in schema['device']['options']
