from types import SimpleNamespace

import pytest

from endeavour_display.markers import DELETE_ALL, Weed, label_from_text, parse_weeds
from endeavour_display.plants import DEFAULT_PLANTS, load_plants


def marker(ns, id, x=0.0, y=0.0, text='', action=0):
    return SimpleNamespace(ns=ns, id=id, action=action, text=text,
                           pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y)))


def test_a_snapshot_becomes_weeds_with_their_labels_ordered_by_track_id():
    snapshot = [marker('', 0, action=DELETE_ALL),
                marker('weeds', 7, 1.0, 2.0), marker('weed_labels', 7, text='plant2 | ID 7'),
                marker('weeds', 3, -0.5, 0.25), marker('weed_labels', 3, text='plant1 | ID 3')]
    assert parse_weeds(snapshot) == [Weed(3, 'plant1', -0.5, 0.25), Weed(7, 'plant2', 1.0, 2.0)]


def test_a_delete_all_inside_the_array_drops_everything_before_it():
    snapshot = [marker('weeds', 1, 1.0, 1.0), marker('', 0, action=DELETE_ALL), marker('weeds', 2, 2.0, 2.0)]
    assert [w.track_id for w in parse_weeds(snapshot)] == [2]


def test_a_weed_without_a_label_marker_is_unknown_and_other_namespaces_are_ignored():
    snapshot = [marker('weeds', 4, 1.0, 1.0), marker('rover', 1, 9.0, 9.0)]
    assert parse_weeds(snapshot) == [Weed(4, 'unknown', 1.0, 1.0)]
    assert parse_weeds([]) == []


@pytest.mark.parametrize('text, label', [('plant2 | ID 14', 'plant2'), ('  weed  | ID 1', 'weed'), ('', 'unknown')])
def test_label_from_text(text, label):
    assert label_from_text(text) == label


def test_default_plants_are_plant1_to_plant4():
    assert [p['label'] for p in load_plants()] == ['plant1', 'plant2', 'plant3', 'plant4']
    assert load_plants() is not DEFAULT_PLANTS     # callers get their own copy


def test_plants_come_from_a_yaml_file_and_labels_are_checked(tmp_path):
    good = tmp_path / 'plants.yaml'
    good.write_text('plants:\n  - {label: thistle, name: Thistle}\n  - {label: dock}\n')
    assert load_plants(good) == [{'label': 'thistle', 'name': 'Thistle'}, {'label': 'dock', 'name': 'dock'}]

    for text in ('plants:\n  - {label: "a b"}\n', 'plants:\n  - {label: "x,y"}\n', 'plants: []\n'):
        bad = tmp_path / 'bad.yaml'
        bad.write_text(text)
        with pytest.raises(ValueError):
            load_plants(bad)
