from .instance_head import InstanceSegmentationHead
from .semantic_head import SemanticSegmentationHead, VideoSemanticSegmentationHead
from .video_object_segmentation_head import VideoObjectSegmentationHead

__all__ = ["InstanceSegmentationHead", "SemanticSegmentationHead", "VideoSemanticSegmentationHead", "VideoObjectSegmentationHead"]
