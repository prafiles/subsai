#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the four backend adapters: config plumbing and transcribe
result -> SSAFile conversion. The underlying libraries (whisper,
whisper_timestamped, faster-whisper, pywhispercpp) are patched because they
would otherwise download multi-GB weights; every subsai-side code path
(_load_config plumbing, decode-options assembly, SSA event building, time
scaling) runs for real.
"""
import types

import pytest
from pysubs2 import SSAFile

from subsai.models.faster_whisper_model import FasterWhisperModel
from subsai.models.whisper_model import WhisperModel
from subsai.models.whisper_timestamped_model import WhisperTimeStamped
from subsai.models.whispercpp_model import WhisperCppModel

WHISPER_RESULT = {
    'segments': [
        {'id': 0, 'seek': 0, 'start': 0.0, 'end': 2.5, 'text': ' Hello there',
         'tokens': [], 'temperature': 0.0, 'avg_logprob': -0.2,
         'compression_ratio': 1.1, 'no_speech_prob': 0.01},
        {'id': 1, 'seek': 0, 'start': 3.0, 'end': 5.0, 'text': ' Second part',
         'tokens': [], 'temperature': 0.0, 'avg_logprob': -0.2,
         'compression_ratio': 1.1, 'no_speech_prob': 0.01},
    ],
    'language': 'en',
    'text': ' Hello there Second part',
}

WT_RESULT = {
    'segments': [
        {'start': 1.0, 'end': 2.0, 'text': ' First line',
         'words': [{'start': 1.0, 'end': 1.4, 'text': 'first'},
                   {'start': 1.5, 'end': 2.0, 'text': 'line'}]},
        {'start': 3.0, 'end': 4.0, 'text': ' Second line',
         'words': [{'start': 3.0, 'end': 3.5, 'text': 'second'},
                   {'start': 3.6, 'end': 4.0, 'text': 'line'}]},
    ],
    'text': ' First line Second line',
    'language': 'en',
}


class TestWhisperModel:
    def test_defaults(self, monkeypatch):
        captured = {}

        def fake_load_model(name, device=None, download_root=None, in_memory=False):
            captured.update(name=name, device=device,
                            download_root=download_root, in_memory=in_memory)
            return 'MODEL'

        monkeypatch.setattr('subsai.models.whisper_model.whisper.load_model',
                            fake_load_model)
        model = WhisperModel({})
        assert model.model == 'MODEL'
        assert captured == {'name': 'base', 'device': None,
                            'download_root': None, 'in_memory': False}
        assert model.model_name == 'openai/whisper'

    def test_overrides(self, monkeypatch):
        captured = {}
        monkeypatch.setattr('subsai.models.whisper_model.whisper.load_model',
                            lambda **kw: captured.update(kw) or 'MODEL')
        model = WhisperModel({'model_type': 'tiny', 'device': 'cpu',
                              'in_memory': True, 'task': 'translate',
                              'beam_size': 10})
        assert captured['name'] == 'tiny'
        assert captured['device'] == 'cpu'
        assert captured['in_memory'] is True
        assert model.decode_options['task'] == 'translate'
        assert model.decode_options['beam_size'] == 10

    def test_decode_options_exclude_explicit_attrs(self, monkeypatch):
        monkeypatch.setattr('subsai.models.whisper_model.whisper.load_model',
                            lambda **kw: 'MODEL')
        model = WhisperModel({})
        for explicit in ['model_type', 'device', 'download_root', 'in_memory',
                         'verbose', 'temperature']:
            assert explicit not in model.decode_options
        assert 'beam_size' in model.decode_options

    def test_transcribe_builds_ssafile(self, monkeypatch):
        monkeypatch.setattr('subsai.models.whisper_model.whisper.load_model',
                            lambda **kw: 'MODEL')
        monkeypatch.setattr('subsai.models.whisper_model.whisper.load_audio',
                            lambda f: 'AUDIO')
        calls = {}

        class M:
            def transcribe(self, audio, **kwargs):
                calls['audio'] = audio
                calls['kwargs'] = kwargs
                return WHISPER_RESULT

        model = WhisperModel({})
        model.model = M()
        subs = model.transcribe('fake.mp4')
        assert calls['audio'] == 'AUDIO'
        assert isinstance(subs, SSAFile)
        assert len(subs) == 2
        assert subs[0].text == 'Hello there'
        assert subs[0].start == 0 and subs[0].end == 2500
        assert subs[1].start == 3000 and subs[1].end == 5000


class TestWhisperTimeStamped:
    def test_defaults(self, monkeypatch):
        captured = {}

        def fake_load_model(name=None, device=None, download_root=None, in_memory=False):
            captured.update(name=name, device=device)
            return 'MODEL'

        monkeypatch.setattr('subsai.models.whisper_timestamped_model.whisper_timestamped.load_model',
                            fake_load_model)
        model = WhisperTimeStamped({})
        assert captured == {'name': 'base', 'device': None}
        assert model.segment_type == 'sentence'

    def test_transcribe_sentence(self, monkeypatch):
        monkeypatch.setattr('subsai.models.whisper_timestamped_model.whisper_timestamped.load_model',
                            lambda **kw: 'MODEL')
        monkeypatch.setattr('subsai.models.whisper_timestamped_model.whisper_timestamped.load_audio',
                            lambda f: 'AUDIO')
        monkeypatch.setattr('subsai.models.whisper_timestamped_model.whisper_timestamped.transcribe',
                            lambda model, audio, **kw: WT_RESULT)
        subs = WhisperTimeStamped({}).transcribe('fake.mp4')
        assert len(subs) == 2
        assert subs[0].text == 'First line'
        assert subs[0].start == 1000 and subs[0].end == 2000

    def test_transcribe_word_level(self, monkeypatch):
        monkeypatch.setattr('subsai.models.whisper_timestamped_model.whisper_timestamped.load_model',
                            lambda **kw: 'MODEL')
        monkeypatch.setattr('subsai.models.whisper_timestamped_model.whisper_timestamped.load_audio',
                            lambda f: 'AUDIO')
        monkeypatch.setattr('subsai.models.whisper_timestamped_model.whisper_timestamped.transcribe',
                            lambda model, audio, **kw: WT_RESULT)
        subs = WhisperTimeStamped({'segment_type': 'word'}).transcribe('fake.mp4')
        assert len(subs) == 4
        assert subs[0].text == 'first'
        assert subs[1].start == 1500 and subs[1].end == 2000

    def test_unknown_segment_type_raises(self, monkeypatch):
        monkeypatch.setattr('subsai.models.whisper_timestamped_model.whisper_timestamped.load_model',
                            lambda **kw: 'MODEL')
        monkeypatch.setattr('subsai.models.whisper_timestamped_model.whisper_timestamped.load_audio',
                            lambda f: 'AUDIO')
        monkeypatch.setattr('subsai.models.whisper_timestamped_model.whisper_timestamped.transcribe',
                            lambda model, audio, **kw: WT_RESULT)
        model = WhisperTimeStamped({'segment_type': 'syllable'})
        with pytest.raises(Exception, match='Unknown `segment_type`'):
            model.transcribe('fake.mp4')


class TestFasterWhisperModel:
    def test_load_kwargs(self, monkeypatch):
        captured = {}

        class FakeFW:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr('subsai.models.faster_whisper_model.WhisperModel', FakeFW)
        model = FasterWhisperModel({})
        assert captured['model_size_or_path'] == 'base'
        assert captured['device'] == 'auto'
        assert captured['cpu_threads'] == 0
        assert model.model_name == 'guillaumekln/faster-whisper'

    def test_transcribe_configs_exclude_private(self, monkeypatch):
        monkeypatch.setattr('subsai.models.faster_whisper_model.WhisperModel',
                            lambda **kw: object())
        model = FasterWhisperModel({'language': 'en', 'vad_filter': True})
        for private in ['model_size_or_path', 'device', 'device_index',
                        'compute_type', 'cpu_threads', 'num_workers']:
            assert private not in model.transcribe_configs
        assert model.transcribe_configs['language'] == 'en'
        assert model.transcribe_configs['vad_filter'] is True
        assert model.transcribe_configs['word_timestamps'] is False

    def test_transcribe_segments(self, monkeypatch):
        monkeypatch.setattr('subsai.models.faster_whisper_model.WhisperModel',
                            lambda **kw: object())
        model = FasterWhisperModel({})

        seg1 = types.SimpleNamespace(start=0.0, end=2.0, text=' One ',
                                     words=[])
        seg2 = types.SimpleNamespace(start=2.5, end=4.0, text=' Two ',
                                     words=[])
        info = types.SimpleNamespace(duration=4.0)

        class M:
            def transcribe(self, media_file, **kwargs):
                return iter([seg1, seg2]), info

        model.model = M()
        subs = model.transcribe('fake.mp4')
        assert isinstance(subs, SSAFile)
        assert len(subs) == 2
        assert subs[0].text == 'One'
        assert subs[0].start == 0 and subs[0].end == 2000
        assert subs[1].start == 2500 and subs[1].end == 4000

    def test_transcribe_word_timestamps(self, monkeypatch):
        monkeypatch.setattr('subsai.models.faster_whisper_model.WhisperModel',
                            lambda **kw: object())
        model = FasterWhisperModel({'word_timestamps': True})

        w1 = types.SimpleNamespace(word=' Hello', start=0.1, end=0.5)
        w2 = types.SimpleNamespace(word=' world', start=0.6, end=1.0)
        seg = types.SimpleNamespace(start=0.0, end=1.0, text=' Hello world',
                                    words=[w1, w2])
        # duration > last segment end -> exercises the trailing pbar update
        info = types.SimpleNamespace(duration=2.0)

        class M:
            def transcribe(self, media_file, **kwargs):
                return iter([seg]), info

        model.model = M()
        subs = model.transcribe('fake.mp4')
        assert len(subs) == 2
        assert subs[0].text == 'Hello'
        assert subs[0].start == 100 and subs[0].end == 500
        assert subs[1].text == 'world'


class TestWhisperCppModel:
    def test_none_params_dropped(self, monkeypatch):
        captured = {}

        class FakeModel:
            def __init__(self, model=None, **params):
                captured['model'] = model
                captured['params'] = params

        monkeypatch.setattr('subsai.models.whispercpp_model.Model', FakeModel)
        model = WhisperCppModel({})
        assert captured['model'] == 'base'
        # 'translate' default False is kept, None-valued options dropped
        assert captured['params']['translate'] is False
        assert captured['params']['n_threads'] == 4
        assert model.model_name == 'ggerganov/whisper.cpp'

    def test_transcribe_time_scaling(self, monkeypatch):
        monkeypatch.setattr('subsai.models.whispercpp_model.Model',
                            lambda **kw: object())
        model = WhisperCppModel({})

        class Seg:
            def __init__(self, t0, t1, text):
                self.t0, self.t1, self.text = t0, t1, text

        class M:
            def transcribe(self, media):
                return [Seg(100, 200, ' Alpha '), Seg(250, 400, ' Beta ')]

        model.model = M()
        subs = model.transcribe('fake.wav')
        assert len(subs) == 2
        # whisper.cpp timestamps are in 10 ms units -> ms = t * 10
        assert subs[0].start == 1000 and subs[0].end == 2000
        assert subs[0].text == 'Alpha'
        assert subs[1].start == 2500 and subs[1].end == 4000


class TestWhisperCppNoneParams:
    def test_explicit_none_dropped_from_params(self, monkeypatch):
        captured = {}

        class FakeModel:
            def __init__(self, model=None, **params):
                captured.update(model=model, params=params)

        monkeypatch.setattr('subsai.models.whispercpp_model.Model', FakeModel)
        WhisperCppModel({'language': None, 'n_threads': 8})
        # None values are dropped so pywhispercpp keeps its own defaults
        assert 'language' not in captured['params']
        assert captured['params']['n_threads'] == 8
