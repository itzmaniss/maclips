import json
from pathlib import Path
import pytest
from maclips.review_cli import render_review


def test_import_rejects_invented_timing_before_render(tmp_path):
    words=tmp_path/'words.json'
    words.write_text(json.dumps({'words':[{'word':'hi','start':1,'end':2}]}))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'video':'missing','audio':'missing','transcript':str(words),
        'candidates':[{'rank':1,'start_word':0,'end_word':0,'start':0,'end':2}]}))
    with pytest.raises(ValueError,match='does not match'):
        render_review(manifest)
