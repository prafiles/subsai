#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for subsai.utils (no models, no network)."""
import pytest

from subsai.utils import (
    _load_config,
    available_subs_formats,
    available_translation_models,
    get_available_devices,
)


class TestLoadConfig:
    schema = {
        'alpha': {'type': int, 'description': 'd', 'options': None, 'default': 7},
        'beta': {'type': int, 'description': 'd', 'options': None, 'default': None},
    }

    def test_value_from_config(self):
        assert _load_config('alpha', {'alpha': 3}, self.schema) == 3

    def test_default_when_missing(self):
        assert _load_config('alpha', {}, self.schema) == 7

    def test_explicit_none_is_preserved(self):
        assert _load_config('alpha', {'alpha': None}, self.schema) is None

    def test_none_default(self):
        assert _load_config('beta', {}, self.schema) is None

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            _load_config('gamma', {}, self.schema)


class TestDevices:
    def test_cpu_always_present(self):
        devices = get_available_devices()
        assert isinstance(devices, list)
        assert devices[0] == 'cpu'

    def test_cuda_entries_formatted(self):
        for device in get_available_devices()[1:]:
            assert device.startswith('cuda:')


class TestTranslationModels:
    def test_known_models(self):
        models = available_translation_models()
        assert isinstance(models, list)
        assert 'facebook/m2m100_418M' in models
        assert len(models) == 3


class TestSubsFormats:
    def test_with_extensions(self):
        formats = available_subs_formats()
        assert '.srt' in formats
        assert '.vtt' in formats

    def test_without_extensions(self):
        formats = available_subs_formats(include_extensions=False)
        assert 'srt' in formats
        assert all('.' not in ext for ext in formats)

    def test_same_count_both_modes(self):
        assert len(available_subs_formats()) == len(
            available_subs_formats(include_extensions=False)
        )
