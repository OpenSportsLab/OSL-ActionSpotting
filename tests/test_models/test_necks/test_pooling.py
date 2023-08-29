from snspotting.models.necks import MaxPool, MaxPool_temporally_aware
from snspotting.models.necks import AvgPool, AvgPool_temporally_aware
from snspotting.models.necks import NetVLAD, NetVLAD_temporally_aware
from snspotting.models.necks import NetRVLAD, NetRVLAD_temporally_aware

def test_MaxPool():
    """Tests MaxPool"""
    nb_frames=60
    MaxPool(nb_frames=nb_frames)

def test_MaxPool_temporally_aware():
    """Tests MaxPool_temporally_aware"""
    nb_frames=60
    MaxPool_temporally_aware(nb_frames=nb_frames)


def test_AvgPool():
    """Tests AvgPool"""
    nb_frames=60
    AvgPool(nb_frames=nb_frames)

def test_AvgPool_temporally_aware():
    """Tests AvgPool_temporally_aware"""
    nb_frames=60
    AvgPool_temporally_aware(nb_frames=nb_frames)


def test_NetRVLAD():
    """Tests NetRVLAD"""
    vocab_size=64
    input_dim=512
    NetRVLAD(
        vocab_size=vocab_size, 
        input_dim=input_dim)

def test_NetRVLAD_temporally_aware():
    """Tests NetRVLAD_temporally_aware"""
    vocab_size=64
    input_dim=512
    NetRVLAD_temporally_aware(
        vocab_size=vocab_size, 
        input_dim=input_dim)


def test_NetVLAD():
    """Tests NetVLAD"""
    vocab_size=64
    input_dim=512
    NetVLAD(
        vocab_size=vocab_size, 
        input_dim=input_dim)

def test_NetVLAD_temporally_aware():
    """Tests NetVLAD_temporally_aware"""
    vocab_size=64
    input_dim=512
    NetVLAD_temporally_aware(
        vocab_size=vocab_size, 
        input_dim=input_dim)
