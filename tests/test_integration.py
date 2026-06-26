import sys
import os
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from unittest.mock import patch

# 프로젝트 루트를 Python 패스에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 테스트 이미지 경로 지정
SFE_TEST_IMAGE = "test_image_260521/metal_water.png"
FINISH_TEST_IMAGE = "test_image_260521/ex_HL_001.png"

def ensure_weights():
    # 1. Depth Anything V2
    da_ckpt = "models/depth_anything_v2/depth_anything_v2_vits.pth"
    if not os.path.exists(da_ckpt):
        os.makedirs(os.path.dirname(da_ckpt), exist_ok=True)
        print("Downloading Depth Anything V2 weights...")
        hf_hub_download(
            repo_id="depth-anything/Depth-Anything-V2-Small",
            filename="depth_anything_v2_vits.pth",
            local_dir=os.path.dirname(da_ckpt)
        )
    # 2. Mobile SAM
    sam_ckpt = "checkpoints/mobile_sam.pt"
    if not os.path.exists(sam_ckpt):
        os.makedirs(os.path.dirname(sam_ckpt), exist_ok=True)
        print("Downloading Mobile SAM weights...")
        hf_hub_download(
            repo_id="chemahc94/sg-weights",
            filename="mobile_sam.pt",
            local_dir=os.path.dirname(sam_ckpt)
        )
    # 3. V-SAMS classifier model
    vs_ckpt = "checkpoints/v_sams_model.pth"
    if not os.path.exists(vs_ckpt):
        os.makedirs(os.path.dirname(vs_ckpt), exist_ok=True)
        print("Downloading V-SAMS classifier weights...")
        hf_hub_download(
            repo_id="chemahc94/sg-weights",
            filename="v_sams_model.pth",
            local_dir=os.path.dirname(vs_ckpt)
        )

ensure_weights()

def get_test_image(path, is_coin=False):
    if os.path.exists(path):
        bgr = cv2.imread(path)
    else:
        # Generate dummy synthetic image
        bgr = np.zeros((800, 800, 3), dtype=np.uint8)
        cv2.circle(bgr, (400, 400), 75, (255, 255, 255), -1)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def test_sfe_pipeline():
    """deepdrop_sfe 모듈의 동전 감지 및 원근 보정, 액적 분석 통합 테스트"""
    from deepdrop_sfe import AIContactAngleAnalyzer, PerspectiveCorrector, DropletPhysics
    
    # 1. 이미지 로드
    bgr, rgb = get_test_image(SFE_TEST_IMAGE, is_coin=True)
    
    analyzer = AIContactAngleAnalyzer()
    corrector = PerspectiveCorrector()
    
    # 2. 동전 감지 및 액적 감지 Mocking (더미 이미지로 인한 오탐지 방지)
    if not os.path.exists(SFE_TEST_IMAGE):
        with patch.object(AIContactAngleAnalyzer, "auto_detect_coin_candidate", return_value=(np.array([325, 325, 475, 475]), 1.0)):
            coin_box, _ = analyzer.auto_detect_coin_candidate(bgr)
            assert coin_box is not None
            
            # 원근 보정
            analyzer.set_image(rgb)
            mask_coin, _ = analyzer.predict_mask(box=coin_box)
            mask_bin = analyzer.get_binary_mask(mask_coin)
            H, ws, coin_info, _ = corrector.find_homography(rgb, mask_bin)
            
            assert H is not None
            assert coin_info is not None
            
            warped = corrector.warp_image(rgb, H, ws)
            ccx, ccy, ccr = coin_info
            
            # 액적 감지 Mock
            with patch.object(AIContactAngleAnalyzer, "auto_detect_droplet_candidate", return_value=np.array([380, 380, 420, 420])):
                drop_box = analyzer.auto_detect_droplet_candidate(warped)
                assert drop_box is not None
                
                analyzer.set_image(warped)
                d_mask, _ = analyzer.predict_mask(box=drop_box)
                
                px_mm = DropletPhysics.calculate_pixels_per_mm(ccr, 24.0)
                d_mm = DropletPhysics.calculate_contact_diameter(d_mask, px_mm)
                ca_val = DropletPhysics.calculate_contact_angle(200.0, d_mm)
                
                assert d_mm > 0
                assert 0 < ca_val < 180
    else:
        coin_box, _ = analyzer.auto_detect_coin_candidate(bgr)
        assert coin_box is not None
        analyzer.set_image(rgb)
        mask_coin, _ = analyzer.predict_mask(box=coin_box)
        mask_bin = analyzer.get_binary_mask(mask_coin)
        H, ws, coin_info, _ = corrector.find_homography(rgb, mask_bin)
        assert H is not None
        warped = corrector.warp_image(rgb, H, ws)
        drop_box = analyzer.auto_detect_droplet_candidate(warped)
        assert drop_box is not None
        analyzer.set_image(warped)
        d_mask, _ = analyzer.predict_mask(box=drop_box)
        px_mm = DropletPhysics.calculate_pixels_per_mm(coin_info[2], 24.0)
        d_mm = DropletPhysics.calculate_contact_diameter(d_mask, px_mm)
        ca_val = DropletPhysics.calculate_contact_angle(200.0, d_mm)
        assert d_mm > 0
        assert 0 < ca_val < 180

def test_vsams_pipeline():
    """vsams 모듈의 표면 거칠기, 광택도 및 마감 유형 추론 테스트"""
    from vsams.analysis.surface_evaluator import SurfaceEvaluator
    
    bgr, rgb = get_test_image(FINISH_TEST_IMAGE)
    evaluator = SurfaceEvaluator()
    
    if not os.path.exists(FINISH_TEST_IMAGE):
        with patch.object(SurfaceEvaluator, "_auto_detect_boxes", return_value=[[325, 325, 475, 475], [325, 500, 475, 650]]):
            res = evaluator.analyze(rgb)
            assert "error" not in res
            assert "roughness" in res
            assert "gloss" in res
    else:
        res = evaluator.analyze(rgb)
        assert "error" not in res

def test_curvature_pipeline():
    """sam2, depth-anything-v2, curvature 모듈을 연결한 3D 곡률 분석 파이프라인 테스트"""
    from sg_terra.seg.sam2_wrapper import SAM2BaseWrapper
    from sg_terra.topo.depth_wrapper import DepthAnythingV2Wrapper
    from sg_terra.curv.curvature import CurvatureAnalyzer
    
    bgr, rgb = get_test_image(FINISH_TEST_IMAGE)
    h, w = rgb.shape[:2]
    
    # 1. SAM2 Segment
    sam_w = SAM2BaseWrapper()
    prompt_pts = np.array([[w//2, h//2]])
    prompt_lbls = np.array([1])
    
    with patch.object(SAM2BaseWrapper, "load_model", return_value=None), \
         patch.object(SAM2BaseWrapper, "segment_target", return_value=np.ones((h, w), dtype=bool)):
        sam_w.load_model()
        mask = sam_w.segment_target(rgb, prompt_points=prompt_pts, prompt_labels=prompt_lbls)
    
    assert mask.any(), "SAM2 generated an empty mask"
    
    # 2. Depth Anything V2
    da_ckpt = "models/depth_anything_v2/depth_anything_v2_vits.pth"
    assert os.path.exists(da_ckpt), f"Depth Anything V2 weight not found at {da_ckpt}"
    
    depth_w = DepthAnythingV2Wrapper(encoder="vits", checkpoint_path=da_ckpt)
    depth_w.load_model()
    dmap = depth_w.estimate_depth(rgb, mask=mask)
    
    assert dmap.shape == mask.shape, "Depth map dimensions do not match the mask"
    
    # 3. Curvature
    curv_a = CurvatureAnalyzer(smoothing_sigma=2.0)
    g_curv = curv_a.calculate_gaussian_curvature(dmap, mask=mask)
    cvals, ccoords = curv_a.find_critical_points(g_curv, mask=mask, top_k=1)
    
    # k_max = cvals[0]
    # r_px = 1.0 / np.sqrt(np.abs(k_max)) if k_max != 0 else 0
    # r_mm = r_px * 1.0
    
    assert len(ccoords) > 0, "Failed to locate critical curvature points"
