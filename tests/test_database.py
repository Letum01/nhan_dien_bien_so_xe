from database import RecognitionDatabase


def test_database_add_search_clear(tmp_path):
    db = RecognitionDatabase(str(tmp_path / "history.sqlite3"))
    row_id = db.add("51A00530", "51A00530", "svm", "sample.jpg")
    assert row_id == 1
    assert db.count() == 1
    assert db.search("51A")[0]["source"] == "sample.jpg"
    assert db.search("missing") == []
    db.clear()
    assert db.count() == 0