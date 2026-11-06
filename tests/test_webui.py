#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for webui helper functions. The module imports streamlit in bare
mode (no ScriptRunContext warnings break nothing); only the pure helpers and
the session-state config parser are exercised — the full `webui()` app is a
Streamlit script and is intentionally not rendered here.
"""
import base64
import sys

import pytest
from pysubs2 import SSAFile

webui = pytest.importorskip("subsai.webui", reason="streamlit not installed")


class TestGetKey:
    def test_join(self):
        assert webui._get_key('openai/whisper', 'model_type') == \
            'openai/whisper-model_type'


class TestVttBase64:
    def test_roundtrip(self):
        encoded = webui._vtt_base64('WEBVTT\n\n00:01 --> 00:02\nhi')
        prefix, payload = encoded.split(',', 1)
        assert prefix == 'data:application/octet-stream;base64'
        assert base64.b64decode(payload).decode() == \
            'WEBVTT\n\n00:01 --> 00:02\nhi'

    def test_custom_mime(self):
        assert webui._vtt_base64('x', mime='text/vtt').startswith('data:text/vtt;')


class TestMediaFileBase64:
    def test_empty_path(self):
        out = webui._media_file_base64('')
        assert out[0]['src'].startswith('data:video/mp4;base64,#t=0')

    def test_start_time_appended(self):
        out = webui._media_file_base64('', start_time=42)
        assert out[0]['src'].endswith('#t=42')

    def test_real_file(self, tmp_path):
        f = tmp_path / 'clip.mp4'
        f.write_bytes(b'\x00\x01\x02binary')
        out = webui._media_file_base64(str(f))
        mime, payload = out[0]['src'].split(';base64,')
        # the '#t=<start>' fragment is appended after the payload
        payload = payload.split('#')[0]
        assert base64.b64decode(payload) == b'\x00\x01\x02binary'
        assert 'video/mp4' in mime


class TestSubsDf:
    def test_none_gives_empty_frame(self):
        df = webui._subs_df(None)
        assert list(df.columns) == ['Start time', 'End time', 'Text']
        assert len(df) == 0

    def test_populated(self, sample_subs):
        df = webui._subs_df(sample_subs)
        assert len(df) == 2
        assert df.loc[0, 'Start time'] == '0:00:01.000'
        assert df.loc[0, 'Text'] == 'Hello world'


class TestConfigUi:
    """_config_ui dispatches on the declared type; patch the streamlit
    widgets to record which one is chosen."""

    @pytest.fixture(autouse=True)
    def patch_widgets(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(webui.st, 'text_input',
                            lambda label, **kw: calls.update(widget='text', **{'label': label, 'kw': kw}) or 'T')
        monkeypatch.setattr(webui.st, 'selectbox',
                            lambda label, options, **kw: calls.update(widget='select', **{'label': label, 'kw': kw}) or options[0])
        monkeypatch.setattr(webui.st, 'number_input',
                            lambda label=None, **kw: calls.update(widget='number', **{'label': label, 'kw': kw}) or 1)
        monkeypatch.setattr(webui.st, 'checkbox',
                            lambda label=None, **kw: calls.update(widget='check', **{'label': label, 'kw': kw}) or True)
        self.calls = calls

    def test_str_field(self):
        out = webui._config_ui('lang', 'k', {'type': str, 'description': 'd',
                                             'options': None, 'default': 'en'})
        assert self.calls['widget'] == 'text' and out == 'T'

    def test_list_field(self):
        webui._config_ui('task', 'k', {'type': list, 'description': 'd',
                                       'options': ['transcribe', 'translate'],
                                       'default': 'translate'})
        assert self.calls['widget'] == 'select'
        assert self.calls['kw']['index'] == 1

    def test_float_field(self):
        webui._config_ui('temp', 'k', {'type': float, 'description': 'd',
                                       'options': None, 'default': 0.5})
        assert self.calls['widget'] == 'number'

    def test_float_none_default_uses_text_input(self):
        webui._config_ui('patience', 'k', {'type': float, 'description': 'd',
                                           'options': None, 'default': None})
        assert self.calls['widget'] == 'text'

    def test_bool_field(self):
        webui._config_ui('fp16', 'k', {'type': bool, 'description': 'd',
                                       'options': None, 'default': False})
        assert self.calls['widget'] == 'check'

    def test_unsupported_type_returns_none(self, capsys):
        out = webui._config_ui('weird', 'k', {'type': dict, 'description': 'd',
                                              'options': None, 'default': {}})
        assert out is None
        assert 'does not have a supported UI' in capsys.readouterr().out


class TestGenerateConfigUi:
    def test_calls_config_ui_per_field(self, monkeypatch):
        seen = []
        monkeypatch.setattr(webui, '_config_ui',
                            lambda name, key, config: seen.append((name, key)))
        schema = {
            'a': {'type': str, 'description': '', 'options': None, 'default': ''},
            'b': {'type': bool, 'description': '', 'options': None, 'default': False},
        }
        webui._generate_config_ui('m', schema)
        assert seen == [('a', 'm-a'), ('b', 'm-b')]


class TestGetConfigFromSessionState:
    class Placeholder:
        def __init__(self):
            self.errors = []

        def error(self, msg):
            self.errors.append(msg)

    def test_parsing_rules(self, monkeypatch):
        schema = {
            'lang': {'type': str},
            'ratio': {'type': float},
            'count': {'type': int},
            'flag': {'type': bool},
        }
        state = {'m-lang': '', 'm-ratio': '1.5', 'm-count': '3', 'm-flag': True}
        monkeypatch.setattr(webui.st, 'session_state', state)
        out = webui._get_config_from_session_state('m', schema, self.Placeholder())
        assert out == {'lang': None, 'ratio': 1.5, 'count': 3, 'flag': True}

    def test_literal_none_string_becomes_none(self, monkeypatch):
        schema = {
            'lang': {'type': str},
            'ratio': {'type': float},
            'count': {'type': int},
        }
        monkeypatch.setattr(webui.st, 'session_state',
                            {'m-lang': 'None', 'm-ratio': 'None', 'm-count': 'None'})
        out = webui._get_config_from_session_state('m', schema, self.Placeholder())
        assert out == {'lang': None, 'ratio': None, 'count': None}

    def test_missing_key_skipped(self, monkeypatch):
        schema = {'x': {'type': str}}
        monkeypatch.setattr(webui.st, 'session_state', {})
        out = webui._get_config_from_session_state('m', schema, self.Placeholder())
        assert out == {}

    def test_bad_value_reports_error(self, monkeypatch):
        schema = {'count': {'type': int}}
        monkeypatch.setattr(webui.st, 'session_state', {'m-count': 'abc'})
        ph = self.Placeholder()
        assert webui._get_config_from_session_state('m', schema, ph) is None
        assert 'Problem parsing configs' in ph.errors[0]


class TestCachedWrappers:
    """_create_translation_model / _transcribe are thin @st.cache_resource /
    @st.cache_data wrappers. Clear the cache and patch the underlying Tools /
    SubsAI calls so the wrapper body runs for real."""

    def test_create_translation_model(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(webui.tools, 'create_translation_model',
                            lambda name, model_family=None: sentinel)
        webui._create_translation_model.clear()
        assert webui._create_translation_model('m2m100') is sentinel

    def test_transcribe(self, monkeypatch):
        monkeypatch.setattr(webui.subs_ai, 'create_model',
                            staticmethod(lambda name, model_config=None: 'ADAPTER'))
        monkeypatch.setattr(webui.subs_ai, 'transcribe',
                            staticmethod(lambda media_file, model: f'subs:{model}'))
        webui._transcribe.clear()
        assert webui._transcribe('f.mp4', 'openai/whisper', {}) == 'subs:ADAPTER'


class TestRun:
    def test_bare_mode_relaunches_streamlit(self, monkeypatch):
        captured = {}

        def fake_stcli_main():
            captured['argv'] = list(sys.argv)
            return 0

        monkeypatch.setattr(webui.runtime, 'exists', lambda: False)
        monkeypatch.setattr(webui.stcli, 'main', fake_stcli_main)
        monkeypatch.setattr(sys, 'argv', ['subsai-webui'])
        with pytest.raises(SystemExit) as excinfo:
            webui.run()
        assert excinfo.value.code == 0
        assert captured['argv'][0] == 'streamlit'
        assert 'run' in captured['argv']

    def test_runtime_mode_calls_webui(self, monkeypatch):
        called = {}
        monkeypatch.setattr(webui.runtime, 'exists', lambda: True)
        monkeypatch.setattr(webui, 'webui', lambda: called.update(ok=True))
        webui.run()
        assert called == {'ok': True}


class TestMediaFileMimeFallback:
    def test_unguessable_mime_becomes_none(self, tmp_path):
        # Known bug: guess_type returns (None, None) for unknown extensions
        # without raising, so the except/print branch is dead code and mime
        # becomes None -> the data URL loses its declared type.
        f = tmp_path / 'clip.zzzz'
        f.write_bytes(b'abc')
        out = webui._media_file_base64(str(f), mime='video/mp4')
        assert out[0]['type'] is None
        assert out[0]['src'].startswith('data:None;base64,')
