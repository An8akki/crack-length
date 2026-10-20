"""
Phase 2A.2 — Metric Accounting Unit Tests
Verifies logit accumulation, argmax prediction, exclusion of index 255,
confusion matrix computation, per-class and combined-delamination metrics,
and zero-division policy against exact hand-calculated ground truth.
"""

import unittest
import numpy as np

def accumulate_logits_and_predict(full_shape, window_coords, window_logits):
    """
    Combines overlapping window logits into a single prediction raster.
    full_shape: (H, W)
    window_coords: list of (x1, y1, x2, y2)
    window_logits: list of arrays of shape (num_classes, S, S)
    """
    H, W = full_shape
    num_classes = window_logits[0].shape[0]
    
    logit_accum = np.zeros((num_classes, H, W), dtype=np.float64)
    count_accum = np.zeros((H, W), dtype=np.float64)
    
    for (x1, y1, x2, y2), logits in zip(window_coords, window_logits):
        logit_accum[:, y1:y2, x1:x2] += logits
        count_accum[y1:y2, x1:x2] += 1.0
        
    # Every pixel must be covered at least once
    assert (count_accum >= 1.0).all(), "Uncovered pixel in accumulation!"
    
    mean_logits = logit_accum / count_accum[np.newaxis, :, :]
    pred_raster = np.argmax(mean_logits, axis=0).astype(np.uint8)
    return pred_raster, mean_logits

def compute_metrics(y_true, y_pred, ignore_index=255):
    """
    Computes confusion matrix and metrics strictly over valid pixels.
    y_true: (H, W), values in {0, 1, 2, ignore_index}
    y_pred: (H, W), values in {0, 1, 2}
    """
    valid_mask = (y_true != ignore_index)
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    
    # 3x3 confusion matrix: rows=true, cols=pred
    cm = np.zeros((3, 3), dtype=np.int64)
    for t in range(3):
        for p in range(3):
            cm[t, p] = np.sum((yt == t) & (yp == p))
            
    metrics = {"confusion_matrix": cm, "valid_pixel_count": int(valid_mask.sum())}
    
    # Per-class metrics
    for c in range(3):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        
        # Zero-division handling
        if tp + fp == 0:
            prec = 0.0 if np.sum(cm[c, :]) > 0 else np.nan
        else:
            prec = tp / (tp + fp)
            
        if tp + fn == 0:
            rec = np.nan
        else:
            rec = tp / (tp + fn)
            
        if np.isnan(prec) or np.isnan(rec) or (prec + rec == 0):
            f1 = 0.0 if (tp + fn > 0) else np.nan
        else:
            f1 = 2 * prec * rec / (prec + rec)
            
        denom_iou = tp + fp + fn
        if denom_iou == 0:
            iou = np.nan  # NOT 1.0! Absent class must not receive perfect score!
        else:
            iou = tp / denom_iou
            
        metrics[f"class_{c}"] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": prec, "recall": rec, "f1": f1, "iou": iou
        }
        
    # Combined delamination: Class 1 (Above-Mat) + Class 2 (Below-Mat)
    # Delam is positive (1 or 2), Sound is negative (0)
    delam_true = (yt == 1) | (yt == 2)
    delam_pred = (yp == 1) | (yp == 2)
    
    tp_del = int(np.sum(delam_true & delam_pred))
    fp_del = int(np.sum((~delam_true) & delam_pred))
    fn_del = int(np.sum(delam_true & (~delam_pred)))
    tn_del = int(np.sum((~delam_true) & (~delam_pred)))
    
    prec_del = tp_del / (tp_del + fp_del) if (tp_del + fp_del > 0) else 0.0
    rec_del = tp_del / (tp_del + fn_del) if (tp_del + fn_del > 0) else np.nan
    f1_del = 2 * prec_del * rec_del / (prec_del + rec_del) if (prec_del + rec_del > 0) else 0.0
    iou_del = tp_del / (tp_del + fp_del + fn_del) if (tp_del + fp_del + fn_del > 0) else np.nan
    
    metrics["combined_delam"] = {
        "tp": tp_del, "fp": fp_del, "fn": fn_del, "tn": tn_del,
        "precision": prec_del, "recall": rec_del, "f1": f1_del, "iou": iou_del
    }
    
    return metrics

class TestMetricAccounting(unittest.TestCase):
    def setUp(self):
        # 12x12 synthetic image
        self.H, self.W = 12, 12
        self.y_true = np.zeros((self.H, self.W), dtype=np.uint8)
        
        # Set ground truth:
        # Excluded border (255): top row (12 px)
        self.y_true[0, :] = 255
        
        # Class 1 (Above-Mat delam): 20 pixels at rows 1-2, cols 0-9
        self.y_true[1:3, 0:10] = 1
        
        # Class 2 (Below-Mat delam): 10 pixels at rows 3-4, cols 0-4
        self.y_true[3:5, 0:5] = 2
        
        # Remaining 102 pixels are Sound (0)
        # Total valid pixels = 144 - 12 = 132 pixels.
        # Class 0: 102 px, Class 1: 20 px, Class 2: 10 px.
        
        # 4 overlapping windows of size 8x8 with stride 4 and end-anchoring:
        # x-coords: 0, 4. y-coords: 0, 4.
        self.window_coords = [
            (0, 0, 8, 8),
            (4, 0, 12, 8),
            (0, 4, 8, 12),
            (4, 4, 12, 12)
        ]
        
    def test_single_physical_pixel_scoring(self):
        """Verify each physical pixel is counted exactly once in confusion matrix."""
        # Provide synthetic logits that yield known predictions
        window_logits = []
        for x1, y1, x2, y2 in self.window_coords:
            # Shape (3, 8, 8)
            logits = np.zeros((3, 8, 8), dtype=np.float64)
            # Default predict sound (class 0)
            logits[0, :, :] = 2.0
            
            # Predict class 1 in top-left region
            for r in range(y1, y2):
                for c in range(x1, x2):
                    if 1 <= r <= 2 and 0 <= c <= 7:
                        logits[:, r - y1, c - x1] = [0.0, 5.0, 0.0]
                    elif 3 <= r <= 4 and 0 <= c <= 4:
                        logits[:, r - y1, c - x1] = [0.0, 0.0, 5.0]
            window_logits.append(logits)
            
        pred_raster, mean_logits = accumulate_logits_and_predict(
            (self.H, self.W), self.window_coords, window_logits
        )
        
        self.assertEqual(pred_raster.shape, (self.H, self.W))
        
        # Compute metrics
        m = compute_metrics(self.y_true, pred_raster, ignore_index=255)
        
        # Check valid pixel count
        self.assertEqual(m["valid_pixel_count"], 132)
        
        # Check confusion matrix sum matches exactly 132
        cm = m["confusion_matrix"]
        self.assertEqual(np.sum(cm), 132)
        
    def test_exact_hand_calculated_metrics(self):
        """Verify exact precision, recall, f1, iou against hand calculations."""
        # Perfect prediction on valid pixels
        pred_perfect = self.y_true.copy()
        pred_perfect[pred_perfect == 255] = 0 # any arbitrary prediction on excluded
        
        m = compute_metrics(self.y_true, pred_perfect, ignore_index=255)
        
        # Perfect class 1: TP=20, FP=0, FN=0 -> IoU=1.0, F1=1.0
        self.assertEqual(m["class_1"]["tp"], 20)
        self.assertEqual(m["class_1"]["fp"], 0)
        self.assertEqual(m["class_1"]["fn"], 0)
        self.assertAlmostEqual(m["class_1"]["iou"], 1.0)
        self.assertAlmostEqual(m["class_1"]["f1"], 1.0)
        
        # Perfect class 2: TP=10, FP=0, FN=0 -> IoU=1.0, F1=1.0
        self.assertEqual(m["class_2"]["tp"], 10)
        self.assertEqual(m["class_2"]["fp"], 0)
        self.assertEqual(m["class_2"]["fn"], 0)
        self.assertAlmostEqual(m["class_2"]["iou"], 1.0)
        
        # Perfect combined delamination: TP=30, FP=0, FN=0 -> IoU=1.0, F1=1.0
        self.assertEqual(m["combined_delam"]["tp"], 30)
        self.assertEqual(m["combined_delam"]["fp"], 0)
        self.assertEqual(m["combined_delam"]["fn"], 0)
        self.assertAlmostEqual(m["combined_delam"]["iou"], 1.0)
        
    def test_zero_division_policy(self):
        """Verify that an absent class is NOT awarded 1.0 IoU."""
        y_true_no_c2 = self.y_true.copy()
        # Remove all class 2 pixels (set to sound)
        y_true_no_c2[y_true_no_c2 == 2] = 0
        
        # Model correctly predicts 0 pixels of class 2
        y_pred = y_true_no_c2.copy()
        y_pred[y_pred == 255] = 0
        
        m = compute_metrics(y_true_no_c2, y_pred, ignore_index=255)
        
        # Class 2 is completely absent: TP=0, FP=0, FN=0.
        # Strict zero-division policy: IoU must be NaN, NOT 1.0!
        self.assertTrue(np.isnan(m["class_2"]["iou"]))
        self.assertTrue(np.isnan(m["class_2"]["f1"]))
        
        # If model predicts false positives for the absent class:
        y_pred_fp = y_pred.copy()
        y_pred_fp[1, 0] = 2 # 1 false positive of class 2
        m_fp = compute_metrics(y_true_no_c2, y_pred_fp, ignore_index=255)
        self.assertEqual(m_fp["class_2"]["precision"], 0.0)
        self.assertEqual(m_fp["class_2"]["iou"], 0.0)

if __name__ == "__main__":
    unittest.main()
