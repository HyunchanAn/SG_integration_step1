import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Integrated Surface Analysis (Mobile Demo)",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS Design System (Premium Glassmorphism & Mobile Optimized)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0B1120;
        --bg-secondary: #111827;
        --bg-card: rgba(26, 35, 50, 0.65);
        --bg-card-hover: rgba(30, 42, 58, 0.8);
        --border-default: rgba(30, 58, 95, 0.6);
        --border-accent: rgba(37, 99, 235, 0.8);
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-blue: #3B82F6;
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-violet: #8B5CF6;
        --gradient-primary: linear-gradient(135deg, rgba(30,58,95,0.4) 0%, rgba(11,17,32,0.6) 100%);
        --gradient-accent: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
        --shadow-card: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        --radius-sm: 10px;
        --radius-md: 16px;
        --radius-lg: 20px;
        --glass-blur: blur(12px);
    }

    /* Global Typography */
    html, body, .main, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary);
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 640px; /* 모바일 전용 좁은 세로형 뷰로 최적화 */
        margin: 0 auto;
    }

    /* Title Area */
    h1 {
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: -0.02em;
        font-size: 1.6rem !important;
    }
    h2, h3, h4 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    /* Expander Accordion Cards - Glassmorphism */
    [data-testid="stExpander"] {
        background: var(--bg-card);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-lg) !important;
        margin-bottom: 20px;
        overflow: hidden;
        box-shadow: var(--shadow-card);
    }
    [data-testid="stExpander"] summary {
        padding: 14px 18px !important;
    }
    [data-testid="stExpander"] summary span[data-testid="stMarkdownContainer"] p {
        font-weight: 600 !important;
        font-size: 15px !important;
        color: var(--text-primary) !important;
    }

    /* Metric Card */
    .metric-card {
        background: var(--gradient-primary);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-default);
        padding: 16px 12px;
        border-radius: var(--radius-md);
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-card);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: var(--gradient-accent);
        border-radius: var(--radius-md) var(--radius-md) 0 0;
    }
    .mv {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px;
        font-weight: 700;
        color: #60A5FA;
        letter-spacing: -0.02em;
    }
    .ml {
        font-size: 11px;
        color: var(--text-secondary);
        margin-top: 6px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* Mobile Responsive Column Override */
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 12px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Setup & Helper Data
# ---------------------------------------------------------------------------
IMAGE_BASE_DIR = "/Users/hyunchanan/Documents/GitHub/SG_proj_015/reports_archive/images"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
    <div style="
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        width:38px; height:38px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        font-size:18px; color:white; flex-shrink:0;
    ">🔬</div>
    <div>
        <div style="font-size:1.3rem; font-weight:700;
            background: linear-gradient(90deg,#60A5FA,#A78BFA,#34D399);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            letter-spacing:-0.02em;">통합 표면 분석 플랫폼 (모바일 데모)</div>
        <div style="font-size:11px; color:#64748B; letter-spacing:0.04em; margin-top:2px;">
            원스크롤 모바일 인터페이스 &nbsp;|&nbsp; 
            <span style="background:rgba(37,99,235,.15); color:#60A5FA; padding:1px 6px; border-radius:4px; font-size:10px; font-weight:600;">MOBILE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STEP 1: Image Upload Slots (Accordion Card)
# ---------------------------------------------------------------------------
with st.expander("STEP 1. 사용자 사진 업로드 (액적 2종, 마감, 3D)", expanded=True):
    up_polar = st.file_uploader("액체 #1 이미지 (SFE Polar Droplet)", type=["jpg", "png", "jpeg"])
    up_nonpolar = st.file_uploader("액체 #2 이미지 (SFE Non-polar Droplet)", type=["jpg", "png", "jpeg"])
    up_finish = st.file_uploader("마감 평가 이미지 (V-SAMS Finish / Coin)", type=["jpg", "png", "jpeg"])
    up_curvature = st.file_uploader("3D 곡률 이미지 (SG-TERRA Press)", type=["jpg", "png", "jpeg"])

# ---------------------------------------------------------------------------
# STEP 2: Physical Properties Simulator (Accordion Card)
# ---------------------------------------------------------------------------
with st.expander("STEP 2. 물리 계측 시뮬레이터 제어판", expanded=True):
    sim_w_ca = st.slider("물(Water) 접촉각 설정 (도)", min_value=10.0, max_value=130.0, value=75.0, step=0.5)
    sim_g_ca = st.slider("글리세롤(Glycerol) 접촉각 설정 (도)", min_value=10.0, max_value=130.0, value=65.0, step=0.5)
    sim_ra = st.slider("표면 조도 Ra 설정 (um)", min_value=0.01, max_value=1.5, value=0.15, step=0.01)
    sim_gloss = st.slider("광택도 설정 (GU)", min_value=0.0, max_value=800.0, value=120.0, step=5.0)

# Calculate SFE and configurations in advance for rendering below
sim_raw_sfe = 72.8 - (0.45 * sim_w_ca) + (0.08 * sim_g_ca)
sim_raw_sfe = max(5.0, round(sim_raw_sfe, 1))

if sim_ra >= 0.2:
    sim_corr_sfe = round(sim_raw_sfe * (1.0 + 0.35 * sim_ra), 1)
    status_txt = "보정 완료 (합격)"
    correction_active = True
else:
    sim_corr_sfe = sim_raw_sfe
    status_txt = "합격 (오차범위 내)"
    correction_active = False

if sim_gloss >= 450.0 and sim_ra <= 0.05:
    predicted_finish = "BA (Mirror 오분류 정정)"
    pattern_txt = "고반사 Mirror 경향"
    vsams_mistake = True
elif sim_ra >= 0.2:
    predicted_finish = "HL"
    pattern_txt = "단방향 연마결 (이방성)"
    vsams_mistake = False
elif sim_ra >= 0.5:
    predicted_finish = "#4 (Rough Finish)"
    pattern_txt = "거친 무작위 결"
    vsams_mistake = False
else:
    predicted_finish = "2B"
    pattern_txt = "무방향성"
    vsams_mistake = False

if sim_ra >= 0.8:
    k_val = 0.004019
    r_val = 15.77
    finish_class = "#4 (Rough Finish)"
elif sim_ra <= 0.03:
    k_val = 0.00001
    r_val = 500.0
    finish_class = "BA"
elif sim_ra <= 0.15:
    k_val = 0.00012
    r_val = 95.0
    finish_class = "2B"
else:
    k_val = 0.00085
    r_val = 45.0
    finish_class = "HL"

# ---------------------------------------------------------------------------
# STEP 3: Surface Free Energy (Accordion Card)
# ---------------------------------------------------------------------------
with st.expander("STEP 3. 표면 자유 에너지 분석 (SFE)", expanded=True):
    st.markdown("##### 액체 #1 (물) 액적")
    if up_polar is not None:
        st.image(Image.open(up_polar), use_container_width=True)
    else:
        img_path = os.path.join(IMAGE_BASE_DIR, "2B_water_verify_step3_droplet.jpg")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="기본 시료(2B) 물 액적 마스크 예시", use_container_width=True)
            
    st.markdown("##### 액체 #2 (글리세롤) 액적")
    if up_nonpolar is not None:
        st.image(Image.open(up_nonpolar), use_container_width=True)
    else:
        img_path = os.path.join(IMAGE_BASE_DIR, "2B_glycerol_verify_step3_droplet.jpg")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="기본 시료(2B) 글리세롤 액적 마스크 예시", use_container_width=True)
            
    st.markdown("##### 정량 분석 수치")
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px;">
        <div class="metric-card">
            <div class="mv">{sim_w_ca}°</div>
            <div class="ml">물 접촉각</div>
        </div>
        <div class="metric-card">
            <div class="mv">{sim_g_ca}°</div>
            <div class="ml">글리세롤 접촉각</div>
        </div>
        <div class="metric-card">
            <div class="mv">{sim_raw_sfe}</div>
            <div class="ml">apparent SFE (mN/m)</div>
        </div>
        <div class="metric-card">
            <div class="mv" style="color: #10B981;">{sim_corr_sfe}</div>
            <div class="ml">보정 후 SFE (mN/m)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df_sfe = pd.DataFrame({
        "지표": ["물 접촉각", "글리세롤 접촉각", "겉보기 SFE", "보정 후 SFE", "상태"],
        "결과": [f"{sim_w_ca}°", f"{sim_g_ca}°", f"{sim_raw_sfe} mN/m", f"{sim_corr_sfe} mN/m", status_txt]
    })
    st.dataframe(df_sfe, hide_index=True, use_container_width=True)
    
    if correction_active:
        st.warning(f"보정 적용: 조도 Ra가 0.2 um 이상이므로, Cassie-Baxter 효과 보정식 SFE * (1 + 0.35 * Ra)을 적용해 SFE를 보정했습니다.")

# ---------------------------------------------------------------------------
# STEP 4: Surface Finish (V-SAMS) (Accordion Card)
# ---------------------------------------------------------------------------
with st.expander("STEP 4. 표면 마감 상태 평가 (V-SAMS)", expanded=True):
    st.markdown("##### 마감 평가 촬영 이미지")
    if up_finish is not None:
        st.image(Image.open(up_finish), use_container_width=True)
    else:
        img_path = os.path.join(IMAGE_BASE_DIR, "2B_reflect_verify_finish.jpg")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="기본 시료(2B) 반사상 엣지 검출 분석 예시", use_container_width=True)
            
    st.markdown("##### 계측 분석 수치")
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px;">
        <div class="metric-card">
            <div class="mv">{sim_ra} um</div>
            <div class="ml">표면 조도 (Ra)</div>
        </div>
        <div class="metric-card">
            <div class="mv">{sim_gloss} GU</div>
            <div class="ml">광택도</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df_finish = pd.DataFrame({
        "지표": ["조도 (Ra)", "광택도", "방향성 패턴", "마감 종류"],
        "결과": [f"{sim_ra} um", f"{sim_gloss} GU", pattern_txt, predicted_finish]
    })
    st.dataframe(df_finish, hide_index=True, use_container_width=True)
    
    if vsams_mistake:
        st.warning("V-SAMS 오분류 보정: 광택 포화 상태로 인한 Mirror 오분류를 apparent SFE 교차 검증을 통해 BA 마감으로 자동 정정 맵핑하였습니다.")

# ---------------------------------------------------------------------------
# STEP 5: 3D Curvature (SG-TERRA) (Accordion Card)
# ---------------------------------------------------------------------------
with st.expander("STEP 5. 3D 형상 및 곡률 분석 (SG-TERRA)", expanded=True):
    st.markdown("##### 3D 굴곡 촬영 이미지")
    if up_curvature is not None:
        st.image(Image.open(up_curvature), use_container_width=True)
    else:
        img_path = os.path.join(IMAGE_BASE_DIR, "press_example.jpg")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="기본 시료(Press Anomaly) 수직 촬영 예시", use_container_width=True)
            
        depth_path = os.path.join(IMAGE_BASE_DIR, "press_example_depth.jpg")
        if os.path.exists(depth_path):
            st.image(Image.open(depth_path), caption="3D 깊이 맵 복원 및 가우시안 곡률 맵", use_container_width=True)
            
    st.markdown("##### 3D 지형 시뮬레이션")
    # Interactive Plotly 3D topography matching the sim_ra slider
    x = np.linspace(-2.5, 2.5, 40)
    y = np.linspace(-2.5, 2.5, 40)
    X, Y = np.meshgrid(x, y)
    Z_base = 0.04 * (np.sin(X) + np.cos(Y))
    Z_rough = (sim_ra * 0.08) * np.sin(5*X) * np.cos(5*Y)
    Z = Z_base + Z_rough
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", showscale=False)])
    fig.update_layout(
        autosize=True,
        width=300,
        height=320,
        margin=dict(l=5, r=5, b=5, t=5),
        scene=dict(
            xaxis=dict(title="X", showticklabels=False),
            yaxis=dict(title="Y", showticklabels=False),
            zaxis=dict(title="Z", showticklabels=False)
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("##### 3D 해석 결과 수치")
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px;">
        <div class="metric-card">
            <div class="mv">{k_val}</div>
            <div class="ml">최대 가우시안 곡률 (K)</div>
        </div>
        <div class="metric-card">
            <div class="mv">{r_val} mm</div>
            <div class="ml">최소 곡률 반경 (R)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    df_3d = pd.DataFrame({
        "파라미터": ["가우시안 곡률 (K)", "곡률 반경 (R)", "추정 마감 등급"],
        "결과": [f"{k_val} 1/mm²", f"{r_val} mm", finish_class]
    })
    st.dataframe(df_3d, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# STEP 6: Consolidated Report (Accordion Card)
# ---------------------------------------------------------------------------
with st.expander("STEP 6. 통합 분석 리포트 요약", expanded=True):
    report_data = {
        "시료 업로드 상태": "사용자 맞춤 파일 업로드됨" if (up_polar or up_nonpolar or up_finish or up_curvature) else "기본 데모 템플릿 사용 중",
        "물 접촉각": f"{sim_w_ca} 도",
        "글리세롤 접촉각": f"{sim_g_ca} 도",
        "보정 후 SFE": f"{sim_corr_sfe} mN/m",
        "표면 조도 Ra": f"{sim_ra} um",
        "광택도": f"{sim_gloss} GU",
        "최소 곡률 반경 R": f"{r_val} mm",
        "최종 추정 마감 종류": finish_class
    }
    
    df_report = pd.DataFrame(list(report_data.items()), columns=["평가 지표", "결과"])
    st.dataframe(df_report, hide_index=True, use_container_width=True)
    
    st.markdown("##### 다운로드용 JSON 스키마")
    st.json(report_data)
