#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the main SubsAI/Tools API.

Heavy dependencies (model weights, translation weights) are patched; all
subsai logic under test runs for real.
"""
import shutil
from pathlib import Path

import pysubs2
import pytest
from pysubs2 import SSAFile

from subsai import SubsAI, Tools
from subsai.configs import AVAILABLE_MODELS
from subsai.main import Tools as ToolsFromMain
from subsai.models.abstract_model import AbstractModel
from subsai.utils import available_translation_models


class RecordingAdapter(AbstractModel):
    """Minimal AbstractModel used to exercise SubsAI.transcribe plumbing."""
    model_name = 'recording'

    def __init__(self, model_config):
        super().__init__(model_name=self.model_name, model_config=model_config)
        self.transcribe_calls = []

    def transcribe(self, media_file):
        self.transcribe_calls.append(media_file)
        subs = SSAFile()
        event = pysubs2.SSAEvent(start=0, end=1000)
        event.text = 'fake'
        subs.append(event)
        return subs


class TestSubsAIRegistry:
    def test_available_models_matches_registry(self):
        assert SubsAI.available_models() == list(AVAILABLE_MODELS.keys())

    @pytest.mark.parametrize('model_name', list(AVAILABLE_MODELS))
    def test_model_info_keys(self, model_name):
        info = SubsAI.model_info(model_name)
        assert set(info.keys()) == {'description', 'url'}
        assert info['url'].startswith('http')

    @pytest.mark.parametrize('model_name', list(AVAILABLE_MODELS))
    def test_config_schema_is_the_class_schema(self, model_name):
        schema = SubsAI.config_schema(model_name)
        assert schema is AVAILABLE_MODELS[model_name]['class'].config_schema

    def test_unknown_model_raises_keyerror(self):
        for lookup in (SubsAI.model_info, SubsAI.config_schema, SubsAI.create_model):
            with pytest.raises(KeyError):
                lookup('no-such/model')


class TestCreateModel:
    def test_forwards_name_and_config(self, monkeypatch):
        monkeypatch.setitem(AVAILABLE_MODELS, 'fake/model',
                            {'class': RecordingAdapter, 'description': '',
                             'url': '', 'config_schema': {}})
        model = SubsAI.create_model('fake/model', {'a': 1})
        assert isinstance(model, RecordingAdapter)
        assert model.model_config == {'a': 1}

    def test_default_config_is_empty_dict(self, monkeypatch):
        monkeypatch.setitem(AVAILABLE_MODELS, 'fake/model',
                            {'class': RecordingAdapter, 'description': '',
                             'url': '', 'config_schema': {}})
        model = SubsAI.create_model('fake/model')
        assert model.model_config == {}


class TestTranscribe:
    def test_accepts_model_instance(self, tmp_path):
        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        adapter = RecordingAdapter({})
        subs = SubsAI.transcribe(str(media), adapter)
        assert isinstance(subs, SSAFile)
        # subsai resolves the path to absolute before delegating
        assert adapter.transcribe_calls == [str(media.resolve())]

    def test_accepts_model_name(self, monkeypatch, tmp_path):
        monkeypatch.setitem(AVAILABLE_MODELS, 'fake/model',
                            {'class': RecordingAdapter, 'description': '',
                             'url': '', 'config_schema': {}})
        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        subs = SubsAI.transcribe(str(media), 'fake/model', {'k': 'v'})
        assert isinstance(subs, SSAFile)


class TestToolsTranslation:
    def test_available_translation_models_delegates(self):
        assert Tools.available_translation_models() == available_translation_models()

    def test_create_translation_model_forwards_args(self, monkeypatch):
        captured = {}

        class FakeMT:
            def __init__(self, model_or_path=None, model_family=None):
                captured['model_or_path'] = model_or_path
                captured['model_family'] = model_family

        monkeypatch.setattr('subsai.main.TranslationModel', FakeMT)
        mt = Tools.create_translation_model('facebook/m2m100_418M', 'm2m100')
        assert isinstance(mt, FakeMT)
        assert captured == {'model_or_path': 'facebook/m2m100_418M',
                            'model_family': 'm2m100'}

    def test_available_languages_from_instance(self):
        class FakeMT:
            def available_languages(self):
                return ['en', 'ar', 'fr']

        assert Tools.available_translation_languages(FakeMT()) == ['en', 'ar', 'fr']

    def test_available_languages_from_name(self, monkeypatch):
        class FakeMT:
            def available_languages(self):
                return ['en']

        monkeypatch.setattr(Tools, 'create_translation_model',
                            staticmethod(lambda name=None, model_family=None: FakeMT()))
        assert Tools.available_translation_languages('m2m100') == ['en']


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text, source, target, batch_size=None, verbose=None):
        self.calls.append({'text': text, 'source': source, 'target': target,
                           'batch_size': batch_size, 'verbose': verbose})
        return f'[{target}]{text}'


class TestToolsTranslate:
    def test_structure_preserved_and_text_translated(self, sample_subs):
        fake = FakeTranslator()
        out = Tools.translate(sample_subs, 'English', 'Arabic', model=fake)
        assert isinstance(out, SSAFile)
        assert len(out) == len(sample_subs)
        assert [e.start for e in out] == [e.start for e in sample_subs]
        assert [e.end for e in out] == [e.end for e in sample_subs]
        assert out[0].text == '[Arabic]Hello world'

    def test_default_batch_and_verbose(self, sample_subs):
        fake = FakeTranslator()
        Tools.translate(sample_subs, 'en', 'ar', model=fake)
        assert all(c['batch_size'] == 32 and c['verbose'] is False
                   for c in fake.calls)

    def test_translation_configs_forwarded(self, sample_subs):
        fake = FakeTranslator()
        Tools.translate(sample_subs, 'en', 'ar', model=fake,
                        translation_configs={'batch_size': 4, 'verbose': True})
        assert all(c['batch_size'] == 4 and c['verbose'] is True
                   for c in fake.calls)

    def test_model_name_branch_uses_create(self, monkeypatch, sample_subs):
        fake = FakeTranslator()
        monkeypatch.setattr(Tools, 'create_translation_model',
                            staticmethod(lambda model_name='m2m100',
                                         model_family=None: fake))
        out = Tools.translate(sample_subs, 'en', 'ar', model='m2m100')
        assert len(out) == 2

    def test_empty_subs(self):
        fake = FakeTranslator()
        out = Tools.translate(SSAFile(), 'en', 'ar', model=fake)
        assert len(out) == 0
        assert fake.calls == []


class TestToolsAutoSync:
    def test_returns_ssafile_and_cleans_tempfiles(self, sample_subs, tiny_wav, tmp_path):
        before = set(tmp_path.iterdir())
        synced = Tools.auto_sync(sample_subs, tiny_wav)
        assert isinstance(synced, SSAFile)
        assert len(synced) == len(sample_subs)
        assert [e.text for e in synced] == [e.text for e in sample_subs]
        assert set(tmp_path.iterdir()) == before

    def test_bool_and_none_kwargs_are_accepted(self, sample_subs, tiny_wav):
        synced = Tools.auto_sync(sample_subs, tiny_wav,
                                 gss=True, serialize_speech=False,
                                 suppress_output_if_offset_less_than=None)
        assert isinstance(synced, SSAFile)

    def test_value_kwargs_forwarded(self, sample_subs, tiny_wav):
        # ffsubsync args are hyphenated; subsai passes kwargs through verbatim
        synced = Tools.auto_sync(sample_subs, tiny_wav,
                                 **{'max-offset-seconds': 30})
        assert isinstance(synced, SSAFile)


class TestToolsMergeSubs:
    @pytest.fixture
    def video(self, tiny_mp4, tmp_path):
        # merge writes next to the input; keep the session fixture pristine
        dest = tmp_path / 'movie.mp4'
        shutil.copy(tiny_mp4, dest)
        return str(dest)

    def test_default_output_name(self, sample_subs, video):
        out = Tools.merge_subs_with_video({'English': sample_subs}, video)
        expected = str((Path(video).parent / 'movie-subs-merged.mp4').resolve())
        assert out == expected
        assert Path(out).exists()

    def test_explicit_output_filename(self, sample_subs, video):
        out = Tools.merge_subs_with_video({'English': sample_subs}, video,
                                         output_filename='custom')
        assert out.endswith('custom.mp4')
        assert Path(out).exists()

    def test_multiple_subtitle_tracks(self, sample_subs, video):
        subs2 = SSAFile.from_string(sample_subs.to_string(format_='srt'), format_='srt')
        out = Tools.merge_subs_with_video({'English': sample_subs,
                                          'Arabic': subs2}, video,
                                         output_filename='two-tracks')
        assert Path(out).exists()

    def test_audio_file_rejected(self, sample_subs, tiny_wav):
        # ffprobe select_streams=v returns no streams for audio -> the
        # [0] index blows up before the codec_type assert is reached
        with pytest.raises((AssertionError, IndexError)):
            Tools.merge_subs_with_video({'English': sample_subs}, tiny_wav)


def test_tools_exported_from_main():
    assert ToolsFromMain is Tools
