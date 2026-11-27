from unittest.mock import patch

import pytest

from hard15_benchmark import check_milestone,aggregate


def test_incomplete_hard15_never_runs_partial_benchmark():
    protocol={'plate_ids':[str(i) for i in range(15)]}
    completed=[{'image_id':str(i)} for i in range(10)]
    with patch('hard15_benchmark.load_hard_queue',return_value={'errors':[],'completed':completed}):
        with pytest.raises(ValueError,match='5 of frozen'):check_milestone(protocol,object(),object())


def test_complete_queue_order_not_database_order():
    protocol={'plate_ids':['a','b']}
    with patch('hard15_benchmark.load_hard_queue',return_value={'errors':[],'completed':[{'image_id':'b'},{'image_id':'a'}]}):
        assert [r['image_id'] for r in check_milestone(protocol,None,None)]==['a','b']