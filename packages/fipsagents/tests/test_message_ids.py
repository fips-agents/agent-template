"""Tests for stable message ID generation and stamping."""
import time
from fipsagents.baseagent.agent import _generate_message_id, _stamp_message_id


def test_generate_message_id_format():
    mid = _generate_message_id()
    assert mid.startswith("msg_")
    parts = mid.split("_")
    assert len(parts) == 3  # msg, timestamp_hex, random_hex
    # Timestamp part should be 12 hex chars
    assert len(parts[1]) == 12
    # Random part should be 12 hex chars
    assert len(parts[2]) == 12


def test_generate_message_id_sortable():
    id1 = _generate_message_id()
    time.sleep(0.002)  # ensure different millisecond
    id2 = _generate_message_id()
    assert id1 < id2, f"IDs should be sortable: {id1} < {id2}"


def test_stamp_message_id_adds_id():
    msg = {"role": "user", "content": "hello"}
    result = _stamp_message_id(msg)
    assert result is msg  # mutates in place
    assert "id" in msg
    assert msg["id"].startswith("msg_")


def test_stamp_message_id_preserves_existing():
    msg = {"role": "user", "content": "hello", "id": "existing_id"}
    _stamp_message_id(msg)
    assert msg["id"] == "existing_id"


def test_stamp_message_id_idempotent():
    msg = {"role": "user", "content": "hello"}
    _stamp_message_id(msg)
    first_id = msg["id"]
    _stamp_message_id(msg)
    assert msg["id"] == first_id
