from ssd.modeling.detector import build_detection_model

def build_ssd_model(cfg, nID):
    model = build_detection_model(cfg, nID)
    return model
