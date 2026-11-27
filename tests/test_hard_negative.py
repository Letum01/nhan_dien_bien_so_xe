from hard_negative_data import negative, geometry_keep
from hard_negative_row_probe import row_order


def test_mining_excludes_parts_and_outside_plate():
    plate=[10,10,100,80]; truth=[[30,30,10,20]]
    assert negative([12,13,9,10],truth,plate)
    assert not negative([37,31,12,12],truth,plate)
    assert not negative([40,31,12,12],truth,plate)  # two-pixel safety margin
    assert not negative([1,1,9,10],truth,plate)


def test_geometry_does_not_drop_slender_glyph():
    assert geometry_keep({'density':.15,'relative_height':.84,'aspect':.15})
    assert not geometry_keep({'density':.15,'relative_height':.3,'aspect':1.5})


def test_row_order_preserves_boxes_and_sorts_two_rows():
    assert row_order([[30,8,9,20],[8,40,11,20],[10,9,9,20],[28,40,9,20]])==[2,0,1,3]
    assert row_order([[30,8,9,20],[10,9,9,20]])==[0,1]