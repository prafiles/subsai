#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test file the main module

Heavy end-to-end tests: they download real whisper/translation model weights
and require ffmpeg + significant CPU. Marked ``slow``; the default pytest run
excludes them (see [tool.pytest.ini_options] in pyproject.toml). Run with::

    pytest -m slow
"""
import pathlib
import unittest
from unittest import TestCase

import pytest
import pysubs2
from pysubs2 import SSAFile

from subsai import SubsAI, Tools


@pytest.mark.slow
class TestSubsAI(TestCase):
    subs_ai = SubsAI()
    model_name = 'openai/whisper'
    _assets = pathlib.Path(__file__).resolve().parent.parent / 'assets'
    files = [str(_assets / 'video' / 'test1.webm'),
             str(_assets / 'audio' / 'test1.mp3')]

    def test_available_models(self):
        available_models = self.subs_ai.available_models()
        self.assertIsInstance(available_models, list, 'This should return a list')

    def test_model_info(self):
        for model in self.subs_ai.available_models():
            info = self.subs_ai.model_info(model)
            self.assertIn('description', info, f'no description for model {model}')

    def test_config_schema(self):
        for model in self.subs_ai.available_models():
            schema = self.subs_ai.config_schema(model)
            self.assertIsInstance(schema, dict, 'the config schema needs to be a dict object')
            for config in schema:
                for field in ['type', 'description', 'options', 'default']:
                    self.assertIn(field, schema[config], f'No {field} in the schema of the {model} model')

    def test_create_model(self):
        for model in self.subs_ai.available_models():
            model_instance = self.subs_ai.create_model(model)
            self.assertIsNotNone(model_instance, f'Model {model} should not be None')

    def test_transcribe(self):
        for model in self.subs_ai.available_models():
            model_instance = self.subs_ai.create_model(model)
            for file in self.files:
                # was model_instance.transcribe(file, model_instance):
                # AbstractModel.transcribe takes only media_file (see report)
                subs = self.subs_ai.transcribe(file, model_instance)
                self.assertIsInstance(subs, SSAFile, 'transcribe function should return `pysubs2.SSAFile`')


@pytest.mark.slow
class TestTools(TestCase):
    tools = Tools()
    _assets = pathlib.Path(__file__).resolve().parent.parent / 'assets'
    file = str(_assets / 'video' / 'test1.webm')
    subs_file = str(_assets / 'video' / 'test1.srt')

    @classmethod
    def setUpClass(cls):
        if not pathlib.Path(cls.subs_file).exists():
            raise unittest.SkipTest(f'missing fixture {cls.subs_file}')
        cls.subs = pysubs2.load(cls.subs_file)

    def test_translate(self):
        translated_subs = self.tools.translate(self.subs, 'English', 'Arabic')
        self.assertIsInstance(translated_subs, SSAFile)
        self.assertEqual(len(self.subs), len(translated_subs), 'Translation subs should have the same length as the '
                                                               'original subs')

    def test_auto_sync(self):
        synced_subs = self.tools.auto_sync(self.subs, self.file)
        self.assertIsInstance(synced_subs, SSAFile)


    def test_merge_subs_with_video(self):
        # was Tools.merge_subs_with_video2(...) — no such method exists in
        # subsai.main (see report); corrected to the real API name
        Tools.merge_subs_with_video({'English': self.subs}, self.file, 'subs-merged')
        in_file = pathlib.Path(self.file)
        self.assertTrue((in_file.parent / f"subs-merged{in_file.suffix}").exists())
