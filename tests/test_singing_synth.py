from core.singing_synth import note_to_hz, parse_score


def test_note_to_hz_a4():
    assert abs(note_to_hz("A4") - 440.0) < 1e-6


def test_parse_score():
    events = parse_score("C4:0.5, D4:1")
    assert len(events) == 2
    assert events[0][1] == 0.5
    assert events[1][1] == 1.0
