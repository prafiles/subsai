#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the CLI helpers and run() wiring (transcription patched)."""
import json
import pathlib

import pysubs2
import pytest
from pysubs2 import SSAFile

from subsai import cli


class FakeAdapter:
    """Stands in for a model instance; SubsAI.transcribe only calls
    .transcribe(media_file) on it."""

    def transcribe(self, media_file):
        subs = SSAFile()
        event = pysubs2.SSAEvent(start=0, end=1000)
        event.text = 'fake transcript'
        subs.append(event)
        return subs


class TestHandleMediaFile:
    def test_plain_paths_resolved(self, tmp_path):
        f = tmp_path / 'a.mp4'
        f.write_bytes(b'x')
        out = cli._handle_media_file([str(f)])
        assert out == [f.resolve()]

    def test_txt_batch_list(self, tmp_path):
        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        other = tmp_path / 'other.mp4'
        other.write_bytes(b'x')
        listing = tmp_path / 'list.txt'
        listing.write_text(f"{media}\n{other}\n")
        out = cli._handle_media_file([str(listing)])
        assert out == [media.resolve(), other.resolve()]

    def test_txt_blank_line_bug(self, tmp_path):
        # Known bug: `line == ''` never matches readlines() output (lines keep
        # their newline), so blank/whitespace-only lines fall through to
        # Path('').resolve() == CWD instead of being skipped.
        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        listing = tmp_path / 'list.txt'
        listing.write_text(f"{media}\n\n")
        out = cli._handle_media_file([str(listing)])
        assert out[0] == media.resolve()
        assert out[1] == pathlib.Path.cwd()

    def test_mixed_args(self, tmp_path):
        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        listing = tmp_path / 'list.txt'
        listing.write_text(f"{media}\n")
        out = cli._handle_media_file([str(media), str(listing)])
        assert out == [media.resolve(), media.resolve()]


class TestHandleConfigs:
    def test_inline_json(self):
        assert cli._handle_configs('{"a": 1}') == {'a': 1}

    def test_empty_json(self):
        assert cli._handle_configs('{}') == {}

    def test_json_file(self, tmp_path):
        cfg = tmp_path / 'cfg.json'
        cfg.write_text(json.dumps({'model_type': 'tiny'}))
        assert cli._handle_configs(str(cfg)) == {'model_type': 'tiny'}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            cli._handle_configs('not json')


def _run_kwargs(**overrides):
    kwargs = dict(media_file_arg=[], model_name='fake/model',
                  model_configs='{}', destination_folder=None,
                  subs_format='srt', translation_model=None,
                  translation_configs='{}', translation_source_lang=None,
                  translation_target_lang=None, output_suffix=None)
    kwargs.update(overrides)
    return kwargs


class TestRun:
    @pytest.fixture(autouse=True)
    def patch_model(self, monkeypatch):
        monkeypatch.setattr(cli.subs_ai, 'create_model',
                            staticmethod(lambda name, config=None: FakeAdapter()))
        monkeypatch.setattr(cli.subs_ai, 'transcribe',
                            staticmethod(lambda f, model: FakeAdapter().transcribe(f)))

    def test_writes_srt_next_to_media(self, tmp_path):
        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        cli.run(**_run_kwargs(media_file_arg=[str(media)]))
        out = tmp_path / 'movie.srt'
        assert out.exists()
        subs = pysubs2.load(str(out))
        assert subs[0].text == 'fake transcript'

    def test_missing_file_skipped(self, tmp_path):
        cli.run(**_run_kwargs(media_file_arg=[str(tmp_path / 'ghost.mp4')]))
        assert not (tmp_path / 'ghost.srt').exists()

    def test_destination_folder_created(self, tmp_path):
        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        dest = tmp_path / 'subs-out'
        cli.run(**_run_kwargs(media_file_arg=[str(media)],
                              destination_folder=str(dest)))
        assert (dest / 'movie.srt').exists()

    def test_output_suffix(self, tmp_path):
        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        cli.run(**_run_kwargs(media_file_arg=[str(media)],
                              output_suffix='en'))
        assert (tmp_path / 'movie-en.srt').exists()

    def test_format_vtt(self, tmp_path):
        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        cli.run(**_run_kwargs(media_file_arg=[str(media)], subs_format='vtt'))
        assert (tmp_path / 'movie.vtt').exists()

    def test_model_configs_json_forwarded(self, tmp_path, monkeypatch):
        captured = {}

        def fake_create(name, config=None):
            captured['name'] = name
            captured['config'] = config
            return FakeAdapter()

        monkeypatch.setattr(cli.subs_ai, 'create_model',
                            staticmethod(fake_create))
        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        cli.run(**_run_kwargs(media_file_arg=[str(media)],
                              model_configs='{"model_type": "tiny"}'))
        assert captured == {'name': 'fake/model',
                            'config': {'model_type': 'tiny'}}


class TestMainArgParsing:
    def test_version_flag_exits(self, monkeypatch, capsys):
        monkeypatch.setattr('sys.argv', ['subsai', '--version'])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 0
        assert cli.__version__ in capsys.readouterr().out

    def test_main_forwards_args(self, monkeypatch, tmp_path):
        captured = {}
        monkeypatch.setattr(cli, 'run', lambda **kw: captured.update(kw))
        monkeypatch.setattr('sys.argv', ['subsai', str(tmp_path / 'a.mp4'),
                                         '-m', 'openai/whisper',
                                         '-mc', '{"model_type": "tiny"}',
                                         '-f', 'vtt',
                                         '-df', '/tmp/out',
                                         '-os', 'sub'])
        cli.main()
        assert captured['model_name'] == 'openai/whisper'
        assert captured['model_configs'] == '{"model_type": "tiny"}'
        assert captured['subs_format'] == 'vtt'
        assert captured['destination_folder'] == '/tmp/out'
        assert captured['output_suffix'] == 'sub'
        assert captured['media_file_arg'] == [str(tmp_path / 'a.mp4')]

    def test_defaults(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(cli, 'run', lambda **kw: captured.update(kw))
        monkeypatch.setattr('sys.argv', ['subsai', 'file.mp4'])
        cli.main()
        assert captured['model_name'] == cli.SubsAI.available_models()[0]
        assert captured['subs_format'] == 'srt'
        assert captured['translation_model'] is None


class TestRunWithTranslation:
    def test_translation_branch(self, monkeypatch, tmp_path):
        """Exercise the -tm path: create_translation_model is called once and
        its result is reused across files; translate() output is what gets saved."""
        created = []

        class FakeTranslator:
            def translate(self, text, source, target, batch_size=None, verbose=None):
                return f'[{target}]{text}'

        def fake_create(model_name, model_family=None):
            created.append(model_name)
            return FakeTranslator()

        monkeypatch.setattr(cli.tools, 'create_translation_model', fake_create)
        monkeypatch.setattr(cli.subs_ai, 'create_model',
                            staticmethod(lambda name, config=None: FakeAdapter()))
        monkeypatch.setattr(cli.subs_ai, 'transcribe',
                            staticmethod(lambda f, model: FakeAdapter().transcribe(f)))

        media = tmp_path / 'movie.mp4'
        media.write_bytes(b'x')
        cli.run(**_run_kwargs(media_file_arg=[str(media)],
                              translation_model='m2m100',
                              translation_source_lang='en',
                              translation_target_lang='ar',
                              translation_configs='{"batch_size": 8}'))
        assert created == ['m2m100']
        subs = pysubs2.load(str(tmp_path / 'movie.srt'))
        assert subs[0].text == '[ar]fake transcript'
