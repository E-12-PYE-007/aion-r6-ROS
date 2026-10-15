"""The plants offered as tickboxes (config/plants.yaml)."""
import re
from pathlib import Path

import yaml

DEFAULT_PLANTS = [{'label': f'plant{i}', 'name': f'Plant {i}'} for i in range(1, 5)]
LABEL = re.compile(r'^[A-Za-z0-9_\-]+$')   # becomes part of a launch argument, so keep it plain


def load_plants(path=None):
    """[{'label', 'name'}] from a yaml file, or the defaults when no file is given."""
    if not path:
        return [dict(p) for p in DEFAULT_PLANTS]
    entries = yaml.safe_load(Path(path).read_text()).get('plants', [])
    plants = []
    for entry in entries:
        label = str(entry['label'])
        if not LABEL.match(label):
            raise ValueError(f'plant label {label!r} may only contain letters, digits, _ and -')
        plants.append({'label': label, 'name': str(entry.get('name', label))})
    if not plants:
        raise ValueError(f'{path} lists no plants')
    return plants
