"""Tests for client dropout and straggler simulation."""

import pytest
from src.reliability.dropout import DropoutSimulator
from src.reliability.stragglers import StragglerSimulator


def test_dropout_simulator():
    clients = [f"C{i+1}" for i in range(10)]
    sim = DropoutSimulator(dropout_prob=0.3, min_participants=3, seed=42)
    active, dropped = sim.filter_active_clients(clients)

    assert len(active) >= 3
    assert len(active) + len(dropped) == 10
    assert set(active).isdisjoint(set(dropped))


def test_dropout_zero_prob():
    clients = [f"C{i+1}" for i in range(10)]
    sim = DropoutSimulator(dropout_prob=0.0, seed=42)
    active, dropped = sim.filter_active_clients(clients)
    assert len(active) == 10
    assert len(dropped) == 0


def test_straggler_simulator():
    clients = [f"C{i+1}" for i in range(10)]
    sim = StragglerSimulator(straggler_prob=0.2, simulated_delay_sec=0.001, seed=42)
    on_time, delayed = sim.evaluate_stragglers(clients)

    assert len(on_time) + len(delayed) == 10
    assert set(on_time).isdisjoint(set(delayed))
