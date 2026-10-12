from __future__ import annotations

from scam_detector.images import PhotoHash, PhotoIndex, dup_stats, hash_image
from scam_detector.synth_images import near_duplicate, render_room


def _save(img, path):
    img.save(path)
    return path


def test_near_duplicate_pair_is_close_and_different_pair_is_far(tmp_path):
    orig = _save(render_room(11), tmp_path / "a.png")
    dup = _save(near_duplicate(render_room(11), seed=11), tmp_path / "a_dup.jpg")
    other = _save(render_room(21), tmp_path / "b.png")

    ha, hd, hb = hash_image(orig), hash_image(dup), hash_image(other)
    assert max(ha.distance(hd)) <= 10
    assert max(ha.distance(hb)) > 10


def test_photo_index_finds_duplicate_across_listings_only(tmp_path):
    orig = _save(render_room(11), tmp_path / "a.png")
    dup = _save(near_duplicate(render_room(11), seed=3), tmp_path / "a_dup.jpg")
    other = _save(render_room(42), tmp_path / "c.png")

    idx = PhotoIndex(threshold=10)
    idx.add_photos("legit", [orig], price_usd=460, district="Yunusobod")
    idx.add_photos("unrelated", [other], price_usd=300, district="Chilonzor")
    assert len(idx) == 2

    matches = idx.query([hash_image(dup)], exclude_listing="scam", photo_names=["a_dup.jpg"])
    assert [m.other_listing_id for m in matches] == ["legit"]
    assert matches[0].distance <= 10

    # объявление не находит само себя
    own = [h for _, h in idx.hashes_for("legit")]
    assert idx.query(own, exclude_listing="legit") == []


def test_same_photos_different_district_and_price_pattern():
    h = PhotoHash(0xFFFF0000FFFF0000, 0x0F0F0F0F0F0F0F0F)
    idx = PhotoIndex(threshold=10)
    idx.add("legit", "p1", h, price_usd=500, district="Yunusobod")
    near = PhotoHash(h.phash ^ 0b111, h.dhash ^ 0b1)
    matches = idx.query([near], exclude_listing="scam")
    stats = dup_stats(matches, price_usd=250, district_key="Mirobod")
    assert stats.dup_photos == 1 and stats.dup_listings == 1
    assert stats.diff_district == 1
    assert stats.max_price_diff == 0.5

    same = dup_stats(matches, price_usd=500, district_key="Yunusobod")
    assert same.diff_district == 0 and same.max_price_diff == 0.0


def test_missing_photo_is_skipped(tmp_path):
    idx = PhotoIndex()
    assert idx.add_photos("x", [tmp_path / "nope.jpg"]) == []
    assert len(idx) == 0
