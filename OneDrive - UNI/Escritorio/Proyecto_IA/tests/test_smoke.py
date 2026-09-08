import pytest

def test_imports():
    import rostro
    import celular
    import events
    assert hasattr(rostro, 'FaceAttention')
    assert hasattr(celular, 'PhoneDetector')
    assert hasattr(events, 'EventRecorder')
