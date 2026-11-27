import pytest
from run_hard13_v2 import check_gate


def test_hard13_requires_every_real_review():
    protocol={'plates':[{'image_id':str(i),'image_path':str(i)} for i in range(13)]}
    with pytest.raises(ValueError,match='missing reviewed'):check_gate(protocol,{})