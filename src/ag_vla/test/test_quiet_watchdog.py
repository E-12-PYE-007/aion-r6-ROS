from ag_vla.quiet_watchdog import QuietWatchdog


def test_never_trips_before_the_first_arrival():
    watchdog = QuietWatchdog(5.0)
    assert not watchdog.expired(0.0)
    assert not watchdog.expired(1000.0)


def test_trips_only_after_the_timeout_and_recovers_on_the_next_arrival():
    watchdog = QuietWatchdog(5.0)
    watchdog.feed(10.0)
    assert not watchdog.expired(10.0)
    assert not watchdog.expired(15.0)    # exactly the timeout is still fine
    assert watchdog.expired(15.01)
    assert watchdog.expired(60.0)

    watchdog.feed(60.0)
    assert not watchdog.expired(60.0)
    assert not watchdog.expired(64.9)
    assert watchdog.expired(65.5)
