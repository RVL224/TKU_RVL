from .ssd_detector import SSDDetector

_DETECTION_META_ARCHITECTURES = {
    "SSDDetector": SSDDetector
}


def build_detection_model(cfg, nID):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    #上面那行等同 meta_arch = SSDDetector(cfg) 
    return meta_arch(cfg, nID)
