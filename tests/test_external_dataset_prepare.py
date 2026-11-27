import numpy as np
import pytest

from external_dataset_prepare import CLASSES, HashIndex, components, crop, mapping, source_config


SOURCE_ORDER = '123456789ABCDEFGHKLMNPSTUVXYZ0'
CONFIG = source_config(f'nc: 30\nnames: {list(SOURCE_ORDER)!r}\n')


@pytest.mark.parametrize('source_id,name', list(enumerate(SOURCE_ORDER)))
def test_all_source_classes_round_trip(source_id, name):
    forward, reverse = mapping(CONFIG)
    assert forward[source_id] == CLASSES.index(name)
    assert reverse[forward[source_id]] == source_id
    assert sorted(set(CLASSES)-set(SOURCE_ORDER)) == list('IJOQRW')


def test_mapping_rejects_duplicate_names_and_duplicate_keys():
    with pytest.raises(ValueError):
        mapping({'nc':30,'names':['1']*30})
    with pytest.raises(ValueError):
        source_config('nc: 30\nnc: 30\nnames: []')


def test_exact_near_and_transitive_quarantine():
    index=HashIndex()
    index.add('protected','same','0000000000000000')
    assert list(index.hits('same','0000000000000001')) == [('protected','sha256',None)]
    assert list(index.hits('other','000000000000003f')) == [('protected','phash_candidate',6)]
    assert not list(index.hits('other','000000000000007f'))
    result=components(['a','b','c','d'],[('a','b'),('b','c')])
    assert result == dict(a='a',b='a',c='a',d='d')


def test_pixel_crop_preserves_source_and_clips_fractions():
    im=np.arange(80,dtype=np.uint8).reshape(8,10)
    roi, box=crop(im,(-1e-7,1.1,4.1,2.7))
    assert box == [0,1,5,3]
    np.testing.assert_array_equal(roi,im[1:4,:5])