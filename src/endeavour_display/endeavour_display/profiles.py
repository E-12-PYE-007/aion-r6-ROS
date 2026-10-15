"""Which network the display bridge and the Jetson stack use to find each other.

On the LAN they find each other through the Fast DDS discovery server on the Jetson. On the Jetson's hotspot
the server is not used: ROS_DISCOVERY_SERVER is unset on every machine and they rely on direct multicast
discovery. The bridge picks one before it joins ROS, because the environment is read when it starts.
"""
from dataclasses import dataclass, field

DEFAULT_SERVER = '10.43.254.23:11811'
MODES = ('auto', 'server', 'multicast')


@dataclass
class Profile:
    name: str                                  # lan, hotspot or as-is
    hosts: list                                # addresses to ssh to, best first
    discovery: str                             # what the bridge tells the Jetson script: server, multicast, or auto
    environment: dict = field(default_factory=dict)   # variables to set for the bridge; a value of None unsets it


def choose_profile(mode, primary, fallback, server, probe, environ):
    """probe(host) -> bool says whether ssh to a host works. In auto mode, if neither answers the environment is
    left exactly as the shell had it, so starting the display from a shell with the settings already unset works."""
    if mode not in MODES:
        raise ValueError(f'discovery mode must be one of {MODES}, not {mode!r}')
    lan = Profile('lan', [primary, fallback], 'server', {'ROS_DISCOVERY_SERVER': server, 'ROS_SUPER_CLIENT': 'TRUE'})
    hotspot = Profile('hotspot', [fallback, primary], 'multicast', {'ROS_DISCOVERY_SERVER': None, 'ROS_SUPER_CLIENT': None})
    if mode == 'server':
        return lan
    if mode == 'multicast':
        return hotspot
    if primary and probe(primary):
        return lan
    if fallback and probe(fallback):
        return hotspot
    return Profile('as-is', [primary, fallback], 'auto', {})
