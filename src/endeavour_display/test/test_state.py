import pytest

from endeavour_display.markers import Weed
from endeavour_display.state import (CAMERA_TIMEOUT_S, DATA_TIMEOUT_S, FLAG_TIMEOUT_S, LINK_TIMEOUT_S,
                                     MESSAGE_SECONDS, DisplayState)

PLANTS = [{'label': 'plant1', 'name': 'Plant 1'}, {'label': 'plant2', 'name': 'Plant 2'}]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def state(clock):
    return DisplayState(PLANTS, clock=clock)


def test_nothing_is_known_before_the_first_contact(state):
    snap = state.snapshot()
    assert snap['connected'] is False and snap['stack'] == 'unknown' and snap['mode'] == 'off'
    assert snap['pose'] is None and snap['weeds'] == [] and snap['can_edit_selection'] is False


def test_connected_while_the_stack_is_stopped_needs_only_a_recent_link(state, clock):
    state.on_link('stopped')
    assert state.snapshot()['connected'] is True and state.snapshot()['stack'] == 'stopped'
    clock.advance(LINK_TIMEOUT_S + 0.1)
    snap = state.snapshot()
    assert snap['connected'] is False and snap['stack'] == 'unknown'


def test_while_running_the_robot_data_must_be_fresh_as_well(state, clock):
    state.on_link('running')
    assert state.snapshot()['connected'] is False        # no data from the stack yet
    state.on_heartbeat()
    assert state.snapshot()['connected'] is True
    clock.advance(DATA_TIMEOUT_S + 0.1)
    state.on_link('running')                             # ssh still works, but the stack has gone quiet
    assert state.snapshot()['connected'] is False
    state.on_odometry(0, 0, 0)                           # odometry alone also counts as data
    assert state.snapshot()['connected'] is True


def test_the_mode_follows_the_cbf_flags_and_recovery_wins(state, clock):
    state.on_link('running')
    assert state.snapshot()['mode'] == 'vla'
    state.on_filtering(True)
    assert state.snapshot()['mode'] == 'cbf'
    state.on_reversing(True)
    assert state.snapshot()['mode'] == 'recovery'
    state.on_reversing(False)
    assert state.snapshot()['mode'] == 'cbf'
    clock.advance(FLAG_TIMEOUT_S + 0.1)                  # a flag nobody refreshes no longer counts
    state.on_link('running')
    assert state.snapshot()['mode'] == 'vla'


def test_mode_is_off_unless_the_stack_runs(state):
    state.on_link('stopped')
    state.on_filtering(True)
    assert state.snapshot()['mode'] == 'off'


def test_weeds_age_until_they_move_and_vanish_when_the_snapshot_drops_them(state, clock):
    state.on_weeds([Weed(1, 'plant1', 1.0, 1.0), Weed(2, 'plant2', 2.0, 2.0)])
    clock.advance(10)
    state.on_weeds([Weed(1, 'plant1', 1.001, 1.0), Weed(2, 'plant2', 2.5, 2.0)])   # 1 barely moved, 2 was seen again
    ages = {w['id']: w['age'] for w in state.snapshot()['weeds']}
    assert ages[1] == pytest.approx(10) and ages[2] == pytest.approx(0)
    state.on_weeds([Weed(2, 'plant2', 2.5, 2.0)])
    assert [w['id'] for w in state.snapshot()['weeds']] == [2]


def test_the_camera_is_alive_only_while_frames_keep_coming(state, clock):
    assert state.snapshot()['camera_alive'] is False
    state.on_camera_frame()
    assert state.snapshot()['camera_alive'] is True
    clock.advance(CAMERA_TIMEOUT_S + 0.1)
    assert state.snapshot()['camera_alive'] is False


def test_a_command_in_flight_overrides_the_stack_and_holds_the_link(state, clock):
    state.on_link('stopped')
    state.begin_command('starting')
    clock.advance(LINK_TIMEOUT_S * 3)                    # a long start must not drop the light
    snap = state.snapshot()
    assert snap['stack'] == 'starting' and snap['connected'] is True and snap['can_edit_selection'] is False
    state.end_command('nvblox did not start')
    assert state.snapshot()['message'] == 'nvblox did not start'
    clock.advance(MESSAGE_SECONDS + 0.1)
    assert state.snapshot()['message'] is None


def test_the_plants_can_only_be_chosen_while_the_stack_is_stopped(state):
    assert state.set_selected(['plant1']) is False       # link not established yet
    state.on_link('stopped')
    assert state.set_selected(['plant2', 'bogus']) is True
    assert state.selected() == ['plant2'] and state.snapshot()['can_edit_selection'] is True
    state.on_link('running')
    assert state.set_selected(['plant1']) is False and state.selected() == ['plant2']


def test_a_new_run_starts_with_an_empty_map_but_a_stop_keeps_the_last_one(state):
    state.on_link('running')
    state.on_weeds([Weed(1, 'plant1', 1.0, 1.0)])
    state.on_link('stopped')
    assert len(state.snapshot()['weeds']) == 1            # the finished run's map stays visible
    state.begin_command('starting')
    assert state.snapshot()['weeds'] == []
    state.on_weeds([Weed(2, 'plant2', 2.0, 2.0)])         # data arrives before the start command returns
    state.on_link('running')
    state.end_command()
    assert [w['id'] for w in state.snapshot()['weeds']] == [2]


def test_a_run_started_elsewhere_clears_the_map_but_connecting_to_a_running_stack_does_not(state):
    state.on_weeds([Weed(1, 'plant1', 1.0, 1.0)])
    state.on_link('running')                              # first contact: the display just started, keep what it has
    assert len(state.snapshot()['weeds']) == 1
    state.on_link('stopped')
    state.on_weeds([Weed(1, 'plant1', 1.0, 1.0)])
    state.on_link('running')                              # stopped, then running without a Start press
    assert state.snapshot()['weeds'] == []
