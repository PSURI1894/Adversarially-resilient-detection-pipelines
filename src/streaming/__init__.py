"""
================================================================================
STREAMING ARCHITECTURE FOR REAL-TIME IDS PIPELINE
================================================================================
"""

from .kafka_producer import FlowProducer
from .kafka_consumer import FlowConsumer
from .inference_service import RealtimeInferenceService
from .feature_store import FeatureStore

__all__ = [
    "FlowProducer",
    "FlowConsumer",
    "RealtimeInferenceService",
    "FeatureStore",
]
