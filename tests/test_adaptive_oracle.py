from adaptive_oracle import oracle_edits


def test_slot_oracle_cannot_create_missing_box_or_delete_extra_for_free():
    assert oracle_edits('AB', [{'A'}, {'B', 'C'}]) == 0
    assert oracle_edits('AB', [{'A'}]) == 1
    assert oracle_edits('AB', [{'A'}, {'X'}, {'B'}]) == 1
    assert oracle_edits('AB', [{'B'}, {'A'}]) == 2