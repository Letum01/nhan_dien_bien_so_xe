# Đánh giá tổng hợp

12 development, 99 ký tự tham chiếu: CNN95/98 matched glyph,8/12 exact,CER4/99; SVM Fusion95/98,8/12,4/99; RF Fusion94/98,7/12,6/99. Không phải test độc lập.

Plate Round2: mAP50 97,15%, precision/recall91,67%. Character Round1: mAP50 99,50% trên GT crop; F1 98,49% trên YOLO crop. Không đánh đồng phạm vi.

Nguồn artifact cục bộ: experiments/yolo_character_round1/comparison/results.json và experiments/hybrid_features_01/results.json. Deep Clean replay12/12 khớp plate box/crop, glyph box/order, scores và chuỗi ba classifier. Không fit hoặc thay model. Không tuyên bố90% tổng quát.
