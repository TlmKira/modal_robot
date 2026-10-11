"""CPU-only control timing, observation semantics and success-window helpers.

No model/simulator imports. These checks do not project or replace model actions.
"""
import math


def validate_action_timing(physics_dt_s, ticks_per_action=15, action_period_s=0.06):
    if not isinstance(ticks_per_action, int) or ticks_per_action < 1:
        raise ValueError('ticks_per_action must be a positive integer')
    if not math.isfinite(physics_dt_s) or physics_dt_s <= 0:
        raise ValueError('physics_dt_s must be finite and positive')
    if not math.isclose(physics_dt_s * ticks_per_action, action_period_s, abs_tol=1e-9):
        raise ValueError('Physics timestep does not match the configured action period')
    return physics_dt_s * ticks_per_action


def summarize_hold(first_tick, last_tick, samples, required_ticks=500, physics_dt_s=0.004):
    """Sample at first_tick is initial state, not a completed time interval."""
    if not math.isfinite(physics_dt_s) or physics_dt_s <= 0 or not isinstance(required_ticks, int) or required_ticks < 1:
        raise ValueError('Hold duration requires positive dt and integer tick count')
    if first_tick is None:
        return dict(completed=False, continuous_official=False, both_contact_seconds=0.0)
    window = [s for s in samples if first_tick < s['tick'] <= first_tick + required_ticks]
    actual_ticks = [s['tick'] for s in window]
    expected = list(range(first_tick + 1, first_tick + required_ticks + 1))
    complete = last_tick >= first_tick + required_ticks and actual_ticks == expected
    return dict(completed=complete,
                continuous_official=complete and all(s['official'] for s in window),
                both_contact_seconds=sum(bool(s['both_contact']) for s in window) * physics_dt_s)


class SuccessEvents:
    """Generic boolean success transitions; caller captures full states at events."""
    def __init__(self):
        self.first_success_tick = None
        self.previous_success = False

    def observe(self, tick, official):
        events = []
        if official and self.first_success_tick is None:
            self.first_success_tick = tick
            events.append('first_official_success')
        if self.first_success_tick is not None and self.previous_success and not official:
            events.append('official_success_lost')
        if self.first_success_tick is not None and not self.previous_success and official and not events:
            events.append('official_success_regained')
        self.previous_success = bool(official)
        return events
