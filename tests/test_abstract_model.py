#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the AbstractModel contract."""
import pytest
from pysubs2 import SSAFile

from subsai.models.abstract_model import AbstractModel


class Concrete(AbstractModel):
    def transcribe(self, media_file):
        return SSAFile()


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        AbstractModel()


def test_stores_name_and_config():
    model = Concrete(model_name='x/y', model_config={'a': 1})
    assert model.model_name == 'x/y'
    assert model.model_config == {'a': 1}


def test_defaults():
    model = Concrete()
    assert model.model_name is None
    assert model.model_config == {}


def test_all_registered_backends_are_subclasses():
    from subsai.configs import AVAILABLE_MODELS
    for entry in AVAILABLE_MODELS.values():
        assert issubclass(entry['class'], AbstractModel)
