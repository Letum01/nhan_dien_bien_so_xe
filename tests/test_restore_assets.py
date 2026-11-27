import hashlib
import json

import pytest

from restore_assets import restore


def test_restore_verifies_before_copy_and_never_overwrites(tmp_path):
    src, dst = tmp_path/'source', tmp_path/'destination'
    src.mkdir()
    dst.mkdir()
    (src/'model.pt').write_bytes(b'model')
    manifest = tmp_path/'manifest.json'
    entry = {'bytes':5, 'sha256':hashlib.sha256(b'model').hexdigest()}
    manifest.write_text(json.dumps({'model.pt':entry}))
    assert restore(manifest,src,dst)==(1,1)
    assert not (dst/'model.pt').exists()
    assert restore(manifest,src,dst,apply=True)==(1,1)
    (dst/'model.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='hash mismatch'):
        restore(manifest,src,dst,apply=True)
    assert (dst/'model.pt').read_bytes()==b'wrong'
    manifest.write_text(json.dumps({'../model.pt':entry}))
    with pytest.raises(ValueError,match='Unsafe'):
        restore(manifest,src,dst,apply=True)