"""Reads the weed geotagger's MarkerArray snapshots.

Each snapshot opens with a DELETEALL marker, then has two markers per confirmed track: a sphere in
namespace 'weeds' (id = track id, position in the odometry frame) and a text marker in 'weed_labels' whose
text is '<class label> | ID <track id>'.
"""
from dataclasses import dataclass

DELETE_ALL = 3   # visualization_msgs/Marker.DELETEALL
LABEL_SEPARATOR = ' | ID '


@dataclass(frozen=True)
class Weed:
    track_id: int
    label: str
    x: float
    y: float


def label_from_text(text):
    return text.split(LABEL_SEPARATOR)[0].strip() or 'unknown'


def parse_weeds(markers):
    """The weeds in one snapshot, ordered by track id. A snapshot is complete, so tracks that are not in it are gone."""
    positions = {}
    labels = {}
    for marker in markers:
        if marker.action == DELETE_ALL:
            positions.clear()
            labels.clear()
        elif marker.ns == 'weeds':
            positions[marker.id] = (marker.pose.position.x, marker.pose.position.y)
        elif marker.ns == 'weed_labels':
            labels[marker.id] = label_from_text(marker.text)
    return [Weed(track_id, labels.get(track_id, 'unknown'), x, y)
            for track_id, (x, y) in sorted(positions.items())]
