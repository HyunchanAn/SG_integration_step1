import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix

# 프로젝트 루트를 Python 패스에 추가
root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

# ==============================================================================
# 1. 메인 앱(app.py) 엔진 로딩 로직 재사용 및 환경 우회
# ==============================================================================
print("Loading engines via app.py...")
import app  # noqa: E402

# app.py 임포트 시점에 제한된 스레드/설정을 오버라이드하여 가속 환경 복구
import multiprocessing  # noqa: E402
if torch.cuda.is_available():
    torch.set_num_threads(max(1, multiprocessing.cpu_count() - 2))
    app.st.session_state["use_fast_mode"] = False
    app.st.session_state["is_cloud"] = False
    print(f"Hardware Acceleration Enabled: CUDA (Threads: {torch.get_num_threads()})")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    torch.set_num_threads(max(1, multiprocessing.cpu_count() - 2))
    app.st.session_state["use_fast_mode"] = False
    app.st.session_state["is_cloud"] = False
    print(f"Hardware Acceleration Enabled: MPS (Threads: {torch.get_num_threads()})")
else:
    print("No hardware acceleration (CUDA/MPS) available. Falling back to CPU.")

# 엔진 변수 매핑
sfe_analyzer = app.sfe_analyzer
sfe_corrector = app.sfe_corrector
vsams_eval = app.vsams_eval
sam2_w = app.sam2_w
depth_w = app.depth_w
curv_a = app.curv_a

# ==============================================================================
# 2. 헬퍼 함수
# ==============================================================================
def load_image(filename, raw_resolution=False):
    path = os.path.join(root_dir, 'evaluation', 'images', str(filename))
    if not os.path.exists(path):
        return None, None
    bgr = cv2.imread(path)
    if bgr is None:
        return None, None
    
    if not raw_resolution:
        # UI 동작과 동일하게 800px 해상도 제한 다운스케일
        h, w = bgr.shape[:2]
        max_size = 800
        if max(h, w) > max_size:
            scale = max_size / float(max(h, w))
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb

def evaluate_sfe(rgb1, rgb2):
    # 단순화된 SFE 평가 플로우 (두 액체 접촉각 측정)
    # 액체 1
    sfe_analyzer.set_image(rgb1)
    coin_box1, _ = sfe_analyzer.auto_detect_coin_candidate(cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR))
    if coin_box1 is None:
        return np.nan, np.nan, np.nan
    mask_coin1, _ = sfe_analyzer.predict_mask(box=coin_box1)
    H1, ws1, cinfo1, _ = sfe_corrector.find_homography(rgb1, sfe_analyzer.get_binary_mask(mask_coin1))
    if H1 is None:
        return np.nan, np.nan, np.nan
    warped1 = sfe_corrector.warp_image(rgb1, H1, ws1)
    drop_box1 = sfe_analyzer.auto_detect_droplet_candidate(warped1)
    if drop_box1 is None:
        return np.nan, np.nan, np.nan
    sfe_analyzer.set_image(warped1)
    d_mask1, _ = sfe_analyzer.predict_mask(box=drop_box1)
    from deepdrop_sfe import DropletPhysics  # noqa: E402
    px_mm1 = DropletPhysics.calculate_pixels_per_mm(cinfo1[2], 24.0)
    ca1 = DropletPhysics.calculate_contact_angle(200.0, DropletPhysics.calculate_contact_diameter(d_mask1, px_mm1))
    
    # 액체 2
    if rgb2 is not None:
        sfe_analyzer.set_image(rgb2)
        coin_box2, _ = sfe_analyzer.auto_detect_coin_candidate(cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR))
        if coin_box2 is None:
            return ca1, np.nan, np.nan
        mask_coin2, _ = sfe_analyzer.predict_mask(box=coin_box2)
        H2, ws2, cinfo2, _ = sfe_corrector.find_homography(rgb2, sfe_analyzer.get_binary_mask(mask_coin2))
        if H2 is None:
            return ca1, np.nan, np.nan
        warped2 = sfe_corrector.warp_image(rgb2, H2, ws2)
        drop_box2 = sfe_analyzer.auto_detect_droplet_candidate(warped2)
        if drop_box2 is None:
            return ca1, np.nan, np.nan
        sfe_analyzer.set_image(warped2)
        d_mask2, _ = sfe_analyzer.predict_mask(box=drop_box2)
        px_mm2 = DropletPhysics.calculate_pixels_per_mm(cinfo2[2], 24.0)
        ca2 = DropletPhysics.calculate_contact_angle(200.0, DropletPhysics.calculate_contact_diameter(d_mask2, px_mm2))
        
        # OWRK SFE 연산
        try:
            sfe_res = DropletPhysics.calculate_owrk_sfe({"Water": ca1}, {"Diiodomethane": ca2})
            sfe_val = sfe_res['sfe_total']
        except Exception:
            sfe_val = np.nan
        return ca1, ca2, sfe_val
    return ca1, np.nan, np.nan

def evaluate_vsams(rgb):
    res = vsams_eval.analyze(rgb)
    if "error" in res:
        return np.nan, np.nan, np.nan
    return res.get("roughness", np.nan), res.get("gloss", np.nan), res.get("predicted_label", np.nan)

def evaluate_3d(rgb):
    h, w = rgb.shape[:2]
    prompt_pts = np.array([[w//2, h//2]])
    prompt_lbls = np.array([1])
    mask = sam2_w.segment_target(rgb, prompt_points=prompt_pts, prompt_labels=prompt_lbls)
    if not mask.any():
        return np.nan
    dmap = depth_w.estimate_depth(rgb, mask=mask)
    g_curv = curv_a.calculate_gaussian_curvature(dmap, mask=mask)
    cvals, ccoords = curv_a.find_critical_points(g_curv, mask=mask, top_k=1)
    if len(cvals) == 0:
        return np.nan
    k_max = cvals[0]
    return 1.0 / np.sqrt(np.abs(k_max)) if k_max != 0 else 0

# ==============================================================================
# 3. 메인 평가 파이프라인
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate SG Integration Accuracy")
    parser.add_argument("--raw-resolution", action="store_true", help="Use raw image resolution instead of 800px downscale")
    parser.add_argument("--csv", default="evaluation/ground_truth.csv", help="Path to ground truth CSV")
    args = parser.parse_args()

    csv_path = os.path.join(root_dir, args.csv)
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    
    results = {
        "sfe": {"y_true_ca1": [], "y_pred_ca1": [], "y_true_ca2": [], "y_pred_ca2": [], "y_true_sfe": [], "y_pred_sfe": []},
        "vsams": {"y_true_ra": [], "y_pred_ra": [], "y_true_gloss": [], "y_pred_gloss": [], "y_true_finish": [], "y_pred_finish": []},
        "curv3d": {"y_true_r": [], "y_pred_r": []}
    }

    # Safe null parsing and slicing per module type
    for _, row in df.iterrows():
        m_type = str(row.get('module_type', '')).upper()
        
        if m_type == 'SFE':
            # Dropna only on essential SFE columns
            if pd.isna(row.get('filename_1')) or pd.isna(row.get('true_contact_angle_1')):
                continue
            
            bgr1, rgb1 = load_image(row['filename_1'], args.raw_resolution)
            if rgb1 is None:
                continue
            
            bgr2, rgb2 = None, None
            if not pd.isna(row.get('filename_2')):
                bgr2, rgb2 = load_image(row['filename_2'], args.raw_resolution)
            
            p_ca1, p_ca2, p_sfe = evaluate_sfe(rgb1, rgb2)
            
            results["sfe"]["y_true_ca1"].append(float(row['true_contact_angle_1']))
            results["sfe"]["y_pred_ca1"].append(p_ca1)
            
            if not pd.isna(row.get('true_contact_angle_2')) and not np.isnan(p_ca2):
                results["sfe"]["y_true_ca2"].append(float(row['true_contact_angle_2']))
                results["sfe"]["y_pred_ca2"].append(p_ca2)
                
            if not pd.isna(row.get('true_sfe')) and not np.isnan(p_sfe):
                results["sfe"]["y_true_sfe"].append(float(row['true_sfe']))
                results["sfe"]["y_pred_sfe"].append(p_sfe)
                
        elif m_type == 'VSAMS':
            if pd.isna(row.get('filename_1')) or (pd.isna(row.get('true_ra')) and pd.isna(row.get('true_gloss')) and pd.isna(row.get('true_finish'))):
                continue
            
            bgr, rgb = load_image(row['filename_1'], args.raw_resolution)
            if rgb is None:
                continue
            
            p_ra, p_gloss, p_finish = evaluate_vsams(rgb)
            
            if not pd.isna(row.get('true_ra')) and not np.isnan(p_ra):
                results["vsams"]["y_true_ra"].append(float(row['true_ra']))
                results["vsams"]["y_pred_ra"].append(p_ra)
            if not pd.isna(row.get('true_gloss')) and not np.isnan(p_gloss):
                results["vsams"]["y_true_gloss"].append(float(row['true_gloss']))
                results["vsams"]["y_pred_gloss"].append(p_gloss)
            if not pd.isna(row.get('true_finish')) and pd.notna(p_finish):
                results["vsams"]["y_true_finish"].append(str(row['true_finish']))
                results["vsams"]["y_pred_finish"].append(str(p_finish))
                
        elif m_type == '3D':
            if pd.isna(row.get('filename_1')) or pd.isna(row.get('true_curvature_r')):
                continue
            
            bgr, rgb = load_image(row['filename_1'], args.raw_resolution)
            if rgb is None:
                continue
            
            p_r = evaluate_3d(rgb)
            if not np.isnan(p_r):
                results["curv3d"]["y_true_r"].append(float(row['true_curvature_r']))
                results["curv3d"]["y_pred_r"].append(p_r)

    # Calculate metrics
    report = {}
    
    def calc_regr(yt, yp):
        if len(yt) < 2:
            return {"MAE": None, "RMSE": None, "R2": None}
        return {
            "MAE": mean_absolute_error(yt, yp),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "R2": r2_score(yt, yp)
        }
    
    report["SFE_CA1"] = calc_regr(results["sfe"]["y_true_ca1"], results["sfe"]["y_pred_ca1"])
    report["SFE_CA2"] = calc_regr(results["sfe"]["y_true_ca2"], results["sfe"]["y_pred_ca2"])
    report["SFE_TOTAL"] = calc_regr(results["sfe"]["y_true_sfe"], results["sfe"]["y_pred_sfe"])
    
    report["VSAMS_Ra"] = calc_regr(results["vsams"]["y_true_ra"], results["vsams"]["y_pred_ra"])
    report["VSAMS_Gloss"] = calc_regr(results["vsams"]["y_true_gloss"], results["vsams"]["y_pred_gloss"])
    
    if len(results["vsams"]["y_true_finish"]) > 0:
        ytf = results["vsams"]["y_true_finish"]
        ypf = results["vsams"]["y_pred_finish"]
        report["VSAMS_Finish"] = {
            "Accuracy": accuracy_score(ytf, ypf),
            "F1_Score_Macro": f1_score(ytf, ypf, average='macro', zero_division=0),
            "Confusion_Matrix": confusion_matrix(ytf, ypf).tolist(),
            "Labels": sorted(list(set(ytf) | set(ypf)))
        }
    
    report["3D_Curvature_R"] = calc_regr(results["curv3d"]["y_true_r"], results["curv3d"]["y_pred_r"])
    
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(json.dumps(report, indent=4))
    
    with open(os.path.join(root_dir, 'evaluation', 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    print("\nReport saved to evaluation/evaluation_report.json")

if __name__ == "__main__":
    main()
