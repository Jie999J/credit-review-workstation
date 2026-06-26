# -*- coding: utf-8 -*-
"""
人工复核工作台 — 对公授信灰色地带决策系统
基于真实模型输出数据，实现"一屏一决策"的高效人工审核流程。
启动：streamlit run app.py
"""
import os, datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="人工复核工作台", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全局样式 — 科技感玻璃质感主题
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""<style>
/* ── 页面背景：纯白 ── */
.stApp { background: #ffffff !important; }

/* ── 顶部留白修复 ── */
.block-container { padding: 1.5rem 2rem 1rem 2rem !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: transparent !important; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }

/* ── 侧边栏深色 ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2740 0%, #141e30 100%) !important;
    min-width: 280px !important; max-width: 280px !important;
    border-right: 1px solid rgba(139,92,246,0.15);
}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stMarkdown *,
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stCaption *
    { color: #d4dff0 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }

/* 侧边栏下拉框 */
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.15) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] div { color: #d4dff0 !important; }
section[data-testid="stSidebar"] [role="listbox"] { background: #1a2740 !important; border: 1px solid rgba(139,92,246,0.3) !important; }
section[data-testid="stSidebar"] [role="option"] { color: #d4dff0 !important; }
section[data-testid="stSidebar"] [role="option"]:hover { background: rgba(139,92,246,0.25) !important; }
section[data-testid="stSidebar"] [aria-selected="true"] { background: rgba(139,92,246,0.35) !important; }
section[data-testid="stSidebar"] .stDownloadButton button {
    background: linear-gradient(135deg, rgba(139,92,246,0.35), rgba(99,102,241,0.35)) !important;
    color: #d4dff0 !important; border: 1px solid rgba(139,92,246,0.4) !important; border-radius: 8px;
}

/* ── 卡片（白底页面版）── */
.glass {
    background: #ffffff;
    border: 1px solid #dce3ec;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    padding: 16px 18px;
}

/* ── 概率仪表 ── */
.prob-gauge { position: relative; height: 20px; border-radius: 10px; overflow: hidden;
    background: linear-gradient(90deg, #fecaca 0%, #fde68a 30%, #bef264 60%, #86efac 100%); margin: 4px 0; }
.prob-indicator { position: absolute; top: -3px; width: 4px; height: 26px;
    background: #18212f; border-radius: 2px; transform: translateX(-50%);
    box-shadow: 0 0 6px rgba(0,0,0,0.3); }
.threshold-line { position: absolute; top: 0; width: 2px; height: 100%; background: rgba(0,0,0,0.35); }

/* ── SHAP 特征条 ── */
.feat-bar { display: flex; align-items: center; gap: 8px; padding: 8px 12px; margin: 4px 0;
    border-radius: 8px; font-size: 13px; background: rgba(255,255,255,0.6);
    backdrop-filter: blur(8px); transition: transform 0.15s; }
.feat-bar:hover { transform: translateX(3px); }
.feat-bar.pos { border-left: 3px solid #16a34a; }
.feat-bar.neg { border-left: 3px solid #dc2626; }
.feat-name { flex: 1; font-weight: 600; color: #1e293b; }
.feat-val { font-weight: 700; min-width: 58px; text-align: right; font-family: 'SF Mono','Consolas',monospace; font-size: 13px; }
.feat-val.pos { color: #16a34a; }
.feat-val.neg { color: #dc2626; }

/* ── 原因标签 ── */
.reason-tag { display: inline-block; padding: 3px 10px; margin: 1px 3px; border-radius: 12px;
    font-size: 11px; font-weight: 600; background: rgba(139,92,246,0.1); color: #7c3aed;
    border: 1px solid rgba(139,92,246,0.2); }

/* ── 反事实 ── */
.cf-card { background: linear-gradient(135deg, rgba(139,92,246,0.08), rgba(99,102,241,0.08));
    border: 1px solid rgba(139,92,246,0.18); border-radius: 10px; padding: 10px 14px;
    font-size: 13px; margin-top: 6px; }

/* ── 不确定性 ── */
.u-row { display: flex; gap: 8px; margin-top: 6px; }
.u-box { flex: 1; padding: 8px; border-radius: 8px; text-align: center;
    background: rgba(255,255,255,0.5); border: 1px solid rgba(0,0,0,0.06); }
.u-box .uv { font-size: 16px; font-weight: 800; color: #334155; }
.u-box .ul { font-size: 10px; color: #94a3b8; }

/* ── 全局 Streamlit 按钮：大 padding 撑高，比 min-height 更可靠 ── */
div.stButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
    padding: 26px 20px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    transition: all 0.15s !important;
    line-height: 1.3 !important;
    white-space: normal !important;
    box-sizing: border-box !important;
}
div.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 16px rgba(0,0,0,0.1) !important; }

/* ── 决策按钮颜色：用 :has() 从标记 span 找到紧邻按钮 ── */
/* 原理：element-container:has(stMarkdownContainer > span[data-dec]) + element-container button */
div:has(> div > [data-dec="approve"]) + div button,
div:has(> [data-dec="approve"]) + div button {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5) !important;
    color: #1f8a57 !important; border: 2px solid #6ee7b7 !important;
}
div:has(> div > [data-dec="approve"]) + div button:hover,
div:has(> [data-dec="approve"]) + div button:hover { border-color: #22c55e !important; box-shadow: 0 6px 20px rgba(34,197,94,0.25) !important; }

div:has(> div > [data-dec="reject"]) + div button,
div:has(> [data-dec="reject"]) + div button {
    background: linear-gradient(135deg, #fef2f2, #fecaca) !important;
    color: #dc2626 !important; border: 2px solid #fca5a5 !important;
}
div:has(> div > [data-dec="reject"]) + div button:hover,
div:has(> [data-dec="reject"]) + div button:hover { border-color: #ef4444 !important; box-shadow: 0 6px 20px rgba(239,68,68,0.25) !important; }

div:has(> div > [data-dec="defer"]) + div button,
div:has(> [data-dec="defer"]) + div button {
    background: linear-gradient(135deg, #fffbeb, #fef3c7) !important;
    color: #b7791f !important; border: 2px solid #fde68a !important;
}
div:has(> div > [data-dec="defer"]) + div button:hover,
div:has(> [data-dec="defer"]) + div button:hover { border-color: #f59e0b !important; box-shadow: 0 6px 20px rgba(245,158,11,0.25) !important; }

/* ── c1 雷达图列：通过 :has(stPlotlyChart) 精准定位，加卡片背景 ── */
[data-testid="column"]:has([data-testid="stPlotlyChart"]) > [data-testid="stVerticalBlock"] {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 1px solid #ddd6fe;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(139,92,246,0.08);
    padding: 14px 16px;
}

/* ── c2 矛盾特征诊断列：纯 HTML 卡片，同色系 ── */
.col-card {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 1px solid #ddd6fe;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(139,92,246,0.08);
    padding: 16px 18px;
    min-height: 400px;
    box-sizing: border-box;
}
.col-card-title { font-size: 14px; font-weight: 700; color: #1e293b; margin-bottom: 10px; }

/* ── 上下区域分隔 ── */
.section-gap { margin: 12px 0 8px 0; border-top: 1px solid rgba(139,92,246,0.1); padding-top: 8px; }
</style>""", unsafe_allow_html=True)


# ━━━━━━━━ 工具函数 ━━━━━━━━
@st.cache_data
def load_data():
    # 同级目录（Streamlit Cloud 部署时 CSV 与 app.py 同放一个仓库根目录）
    p = os.path.join(os.path.dirname(__file__), "gray_zone_analysis.csv")
    df = pd.read_csv(p)
    df = df.rename(columns={df.columns[0]: "cust_id"})
    df["cust_id"] = df["cust_id"].astype(int)
    return df

FNAME = {
    "结算分":"结算评分","存款分":"存款评分","基础健康分":"企业健康评分",
    "存款我行占比":"存款我行占比","missing_pattern_freq":"信息缺失频率",
    "missing_severity":"缺失严重度","missing_ratio":"字段缺失率",
    "小类_te":"行业小类编码","大类_te":"行业大类编码",
    "debt_leverage":"债务杠杆率","asset_efficiency":"资产运营效率",
    "近12个月代发金额值":"代发金额","近12个月开票金额值":"开票金额",
}

def fn(r): return FNAME.get(r, r)

def parse_shap(row):
    fs = str(row.get("shap_top3_features","")).split("|")
    vs = str(row.get("shap_top3_values","")).split("|")
    out = []
    for f,v in zip(fs,vs):
        try: out.append((f.strip(), float(v.strip())))
        except: pass
    return out

def make_radar(row):
    dims = ["信息完整度","经营活跃度","银企关系","财务稳健性","资质信用","规模体量"]
    vals = [row.get(f"radar_{d}",50) for d in dims]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=dims+[dims[0]],
        fill="toself", fillcolor="rgba(139,92,246,0.12)",
        line=dict(color="#8b5cf6",width=2.5), marker=dict(size=5,color="#8b5cf6")))
    fig.add_trace(go.Scatterpolar(r=[50]*7, theta=dims+[dims[0]],
        line=dict(color="#cbd5e1",width=1,dash="dot")))
    fig.update_layout(polar=dict(
        radialaxis=dict(visible=True,range=[0,100],showticklabels=False,gridcolor="rgba(0,0,0,0.06)"),
        angularaxis=dict(gridcolor="rgba(0,0,0,0.06)",linecolor="rgba(0,0,0,0.06)"),bgcolor="rgba(0,0,0,0)"),
        showlegend=False,margin=dict(l=40,r=40,t=10,b=10),height=240,
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(size=11))
    return fig

def pcol(p):
    if p>=.90: return "#16a34a"
    if p>=.733: return "#059669"
    if p>=.583: return "#d97706"
    if p>=.20: return "#dc2626"
    return "#991b1b"

def zcol(z): return {"自动通过":"#16a34a","自动拒绝":"#dc2626","人工复核":"#8b5cf6","风险提示":"#d97706"}.get(z,"#64748b")
def zemoji(z): return {"自动通过":"✅","自动拒绝":"🚫","人工复核":"🔍","风险提示":"⚠️"}.get(z,"")


# ━━━━━━━━ Session State ━━━━━━━━
for k,v in [("idx",0),("decisions",{}),("undo_stack",[]),("zone_filter","人工复核"),("sort_by","gray_score ↓ 灰度优先")]:
    if k not in st.session_state: st.session_state[k] = v

df_all = load_data()

# ━━━━━━━━ 侧边栏 ━━━━━━━━
with st.sidebar:
    st.markdown("### 🏦 人工复核工作台")
    st.caption("AI + 人机协同 · 灰色地带决策系统")
    st.divider()
    st.markdown("""
<div style="font-size:12px;padding:8px 10px;border-radius:8px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.15);margin-bottom:8px;">
<div style="display:flex;align-items:center;gap:6px;margin:2px 0;"><span style="background:#9fe3da;color:#142236;border-radius:50%;width:17px;height:17px;display:inline-flex;align-items:center;justify-content:center;font-size:9px;font-weight:800;">✓</span> 模型预筛</div>
<div style="width:2px;height:5px;margin-left:7px;background:rgba(255,255,255,0.15);"></div>
<div style="display:flex;align-items:center;gap:6px;margin:2px 0;"><span style="background:#a78bfa;color:#142236;border-radius:50%;width:17px;height:17px;display:inline-flex;align-items:center;justify-content:center;font-size:9px;font-weight:800;">2</span> <b style="color:#e0d4ff !important;">人工复核</b></div>
<div style="width:2px;height:5px;margin-left:7px;background:rgba(255,255,255,0.1);"></div>
<div style="display:flex;align-items:center;gap:6px;margin:2px 0;opacity:0.45;"><span style="background:rgba(255,255,255,0.15);border-radius:50%;width:17px;height:17px;display:inline-flex;align-items:center;justify-content:center;font-size:9px;">3</span> 结果反馈</div>
<div style="width:2px;height:5px;margin-left:7px;background:rgba(255,255,255,0.06);"></div>
<div style="display:flex;align-items:center;gap:6px;margin:2px 0;opacity:0.45;"><span style="background:rgba(255,255,255,0.15);border-radius:50%;width:17px;height:17px;display:inline-flex;align-items:center;justify-content:center;font-size:9px;">4</span> 模型迭代</div>
</div>""", unsafe_allow_html=True)

    st.markdown("##### 筛选与排序")
    zl = ["人工复核","风险提示","全部待审","自动通过","自动拒绝"]
    zf = st.selectbox("决策区域", zl, index=zl.index(st.session_state.zone_filter), key="zf")
    if zf != st.session_state.zone_filter: st.session_state.zone_filter = zf; st.session_state.idx = 0

    sl = ["gray_score ↓ 灰度优先","probability ↑ 低概率优先","probability ↓ 高概率优先","missing_ratio ↓ 缺失优先"]
    sb = st.selectbox("排序方式", sl, index=sl.index(st.session_state.sort_by), key="sb")
    if sb != st.session_state.sort_by: st.session_state.sort_by = sb; st.session_state.idx = 0

    st.divider()
    nr = len(df_all[df_all["decision_zone"]=="人工复核"])
    nk = len(df_all[df_all["decision_zone"]=="风险提示"])
    nd = len(st.session_state.decisions)
    ar = len(df_all[df_all["decision_zone"].isin(["自动通过","自动拒绝"])])/len(df_all)*100

    st.markdown(f"""
<div style="display:grid;gap:4px;">
<div style="display:flex;justify-content:space-between;padding:5px 10px;border-radius:6px;background:rgba(139,92,246,0.06);font-size:13px;"><span>待复核</span><span style="font-weight:700;color:#a78bfa;font-size:15px;">{nr}</span></div>
<div style="display:flex;justify-content:space-between;padding:5px 10px;border-radius:6px;background:rgba(245,158,11,0.06);font-size:13px;"><span>风险提示</span><span style="font-weight:700;color:#f59e0b;font-size:15px;">{nk}</span></div>
<div style="display:flex;justify-content:space-between;padding:5px 10px;border-radius:6px;background:rgba(52,211,153,0.06);font-size:13px;"><span>已审核</span><span style="font-weight:700;color:#34d399;font-size:15px;">{nd}</span></div>
<div style="display:flex;justify-content:space-between;padding:5px 10px;border-radius:6px;background:rgba(96,165,250,0.06);font-size:13px;"><span>自动率</span><span style="font-weight:700;color:#60a5fa;font-size:15px;">{ar:.1f}%</span></div>
</div>""", unsafe_allow_html=True)

    st.divider()
    if nd > 0:
        edf = pd.DataFrame([{"客户号":k,**v} for k,v in st.session_state.decisions.items()])
        st.download_button("📥 导出审核记录", edf.to_csv(index=False).encode("utf-8"), "review_decisions.csv", "text/csv", use_container_width=True)

# ━━━━━━━━ 数据筛选 ━━━━━━━━
if st.session_state.zone_filter == "全部待审":
    dq = df_all[df_all["decision_zone"].isin(["人工复核","风险提示"])].copy()
else:
    dq = df_all[df_all["decision_zone"]==st.session_state.zone_filter].copy()

sc = st.session_state.sort_by.split(" ")[0]
dq = dq.sort_values(sc, ascending="↑" in st.session_state.sort_by).reset_index(drop=True)
if len(dq)==0: st.warning("无待审客户"); st.stop()

st.session_state.idx = max(0, min(st.session_state.idx, len(dq)-1))
row = dq.iloc[st.session_state.idx]
cid = int(row["cust_id"]); prob = float(row["probability"]); zone = row["decision_zone"]
gray = float(row["gray_score"]); miss = float(row["missing_ratio"])
reasons = [r.strip() for r in str(row.get("review_reason","")).split("|") if r.strip()] if pd.notna(row.get("review_reason")) else []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 顶栏：客户概况 + 导航 + 撤回
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
h1, h2, h3 = st.columns([5, 1.5, 1.5])

with h1:
    zbg = {"人工复核":"rgba(139,92,246,0.1)","风险提示":"rgba(245,158,11,0.1)","自动通过":"rgba(22,163,74,0.1)","自动拒绝":"rgba(220,38,38,0.1)"}.get(zone,"#f1f5f9")
    rtags = " ".join(f'<span class="reason-tag">{r}</span>' for r in reasons) if reasons else ""
    st.markdown(f"""<div class="glass" style="padding:12px 18px;">
<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:4px;">
    <span style="font-size:20px;font-weight:800;color:#1e293b;">🏢 客户 #{cid}</span>
    <span style="padding:3px 11px;border-radius:12px;font-size:11px;font-weight:700;background:{zbg};color:{zcol(zone)};">{zemoji(zone)} {zone}</span>
    <span style="font-size:12px;color:#94a3b8;">灰度 {gray:.2f} · 缺失 {miss:.0%}</span>
</div>
<div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:32px;font-weight:800;color:{pcol(prob)};line-height:1;">{prob*100:.1f}%</span>
    <div style="flex:1;">
        <div style="font-size:10px;color:#94a3b8;">授信通过概率（阈值 73.3%）</div>
        <div class="prob-gauge"><div class="prob-indicator" style="left:{prob*100}%;"></div><div class="threshold-line" style="left:73.3%;"></div></div>
        <div style="display:flex;justify-content:space-between;font-size:9px;color:#94a3b8;"><span>← 拒绝区</span><span style="color:#8b5cf6;font-weight:700;">灰色地带</span><span>通过区 →</span></div>
    </div>
</div>
{'<div style="margin-top:4px;">🔎 '+rtags+'</div>' if rtags else ''}
</div>""", unsafe_allow_html=True)

with h2:
    st.markdown(f"""<div class="glass" style="text-align:center;padding:10px;">
<div style="font-size:10px;color:#94a3b8;">审核进度</div>
<div style="font-size:28px;font-weight:800;color:#8b5cf6;">{st.session_state.idx+1}<span style="font-size:13px;color:#94a3b8;"> / {len(dq)}</span></div>
</div>""", unsafe_allow_html=True)
    nc1, nc2 = st.columns(2)
    with nc1:
        if st.button("⬅ 上一个", use_container_width=True, disabled=st.session_state.idx==0, key="prev"):
            st.session_state.idx -= 1; st.rerun()
    with nc2:
        if st.button("下一个 ➡", use_container_width=True, disabled=st.session_state.idx>=len(dq)-1, key="next"):
            st.session_state.idx += 1; st.rerun()

with h3:
    # 撤回 + 跳过
    if st.session_state.undo_stack:
        last = st.session_state.undo_stack[-1]
        dcn = {"approve":"批准","reject":"拒绝","defer":"补充材料"}.get(last["decision"],"")
        st.caption(f"上一条：#{last['cust_id']} → {dcn}")
        if st.button("↩ 撤回上一条决策", use_container_width=True, key="undo"):
            u = st.session_state.undo_stack.pop()
            st.session_state.decisions.pop(u["cust_id"], None)
            if st.session_state.idx > 0: st.session_state.idx -= 1
            st.rerun()
    else:
        st.caption("暂无可撤回的决策")

    if st.button("⏭ 跳过当前", use_container_width=True, key="skip"):
        if st.session_state.idx < len(dq)-1: st.session_state.idx += 1
        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 三列内容区：雷达 | 特征 | 决策
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.markdown("**📊 六维风险画像**")
    st.plotly_chart(make_radar(row), use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"""<div class="u-row">
<div class="u-box"><div class="uv">{row['fold_std']:.4f}</div><div class="ul">折间不稳定</div></div>
<div class="u-box"><div class="uv">{row['model_range']:.4f}</div><div class="ul">模型分歧</div></div>
</div>""", unsafe_allow_html=True)

with c2:
    si = parse_shap(row)
    pos = [(f, v) for f, v in si if v > 0]
    neg = [(f, v) for f, v in si if v <= 0]
    pos_html = ""
    if pos:
        pos_html = '<div style="font-size:11px;color:#16a34a;font-weight:700;margin:6px 0 4px 0;">✅ 支持授信</div>'
        for f, v in pos:
            pos_html += f'<div class="feat-bar pos"><span class="feat-name">{fn(f)}</span><span class="feat-val pos">+{v:.4f}</span></div>'
    neg_html = ""
    if neg:
        neg_html = '<div style="font-size:11px;color:#dc2626;font-weight:700;margin:8px 0 4px 0;">❌ 反对授信</div>'
        for f, v in neg:
            neg_html += f'<div class="feat-bar neg"><span class="feat-name">{fn(f)}</span><span class="feat-val neg">{v:.4f}</span></div>'
    cff = row.get("cf_feature", ""); cfd = row.get("cf_delta", 0); cft = row.get("cf_type", "")
    cf_html = ""
    if pd.notna(cff) and cff:
        dr = "提升" if cfd > 0 else "降低"
        cf_html = f"""<div class="cf-card">
<span style="font-weight:700;color:#7c3aed;">🔄 反事实推演</span><br>
若改善「<b>{fn(cff)}</b>」，概率可{dr} <b>{abs(cfd)*100:.1f}%</b>
<div style="font-size:11px;color:#6d28d9;margin-top:3px;">建议行动：{cft}</div></div>"""
    st.markdown(f"""<div class="col-card">
<div class="col-card-title">⚡ 矛盾特征诊断</div>
{pos_html}{neg_html}{cf_html}
</div>""", unsafe_allow_html=True)

with c3:
    already = cid in st.session_state.decisions
    if already:
        pd_dec = st.session_state.decisions[cid]
        dcm = {"approve": "✅ 已批准", "reject": "❌ 已拒绝", "defer": "📎 补充材料"}
        st.success(f'{dcm.get(pd_dec["decision"], "")}（可重新选择覆盖）')

    st.markdown("**✏️ 复核决策**")

    # span[data-dec] 作为相邻兄弟标记，CSS 用 [data-dec="x"] + div 定位后面的 stButton
    decision_made = None
    st.markdown('<span data-dec="approve"></span>', unsafe_allow_html=True)
    if st.button("✅  批准授信", use_container_width=True, key="b_ap"):
        decision_made = "approve"

    st.markdown('<span data-dec="reject"></span>', unsafe_allow_html=True)
    if st.button("❌  拒绝授信", use_container_width=True, key="b_rj"):
        decision_made = "reject"

    st.markdown('<span data-dec="defer"></span>', unsafe_allow_html=True)
    if st.button("📎  补充材料再审", use_container_width=True, key="b_df"):
        decision_made = "defer"

    with st.expander("📝 添加审核理由和反馈（选填）", expanded=False):
        ro = ["支持特征明显优于反对特征", "反对特征可通过增信缓释", "反对特征集中且难以缓释",
              "行业风险过高，审慎原则", "关键数据缺失，无法判断", "需核实异常指标", "需现场尽调后决定"]
        sr = st.multiselect("决策理由", ro, key="r_ms", placeholder="可多选")
        en = st.text_input("补充说明", placeholder="选填", key="n_inp", label_visibility="collapsed")
        fo = ["模型判断准确", "模型误判-实际优质", "模型误判-实际高风险", "数据缺失导致偏差", "特征权重需调整"]
        sf = st.multiselect("反馈标签", fo, key="f_ms", placeholder="反馈标签（模型迭代用）")

    if decision_made:
        rec = {"decision": decision_made,
               "reason": "; ".join(st.session_state.get("r_ms",[])),
               "note": st.session_state.get("n_inp",""),
               "feedback": "; ".join(st.session_state.get("f_ms",[])),
               "model_prob": prob, "gray_score": gray,
               "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        st.session_state.decisions[cid] = rec
        st.session_state.undo_stack.append({"cust_id":cid, **rec})
        if len(st.session_state.undo_stack)>30: st.session_state.undo_stack = st.session_state.undo_stack[-30:]
        if st.session_state.idx < len(dq)-1: st.session_state.idx += 1
        st.rerun()


# ━━━━━━━━ 底部队列 ━━━━━━━━
st.markdown("---")
bc1, bc2 = st.columns([4, 1])
with bc1: st.markdown("**📋 队列预览**")
with bc2:
    jc1, jc2 = st.columns([2,1])
    with jc1: jid = st.number_input("跳转",min_value=int(dq["cust_id"].min()),max_value=int(dq["cust_id"].max()),value=cid,label_visibility="collapsed",key="jmp")
    with jc2:
        if st.button("🔍",key="jb",use_container_width=True):
            m = dq[dq["cust_id"]==jid].index
            if len(m)>0: st.session_state.idx=m[0]; st.rerun()
            else: st.toast(f"#{jid} 不在队列中")

ps = max(0, st.session_state.idx-2); pe = min(len(dq), ps+10)
pv = dq.iloc[ps:pe][["cust_id","probability","decision_zone","gray_score","review_reason","missing_ratio"]].copy()
pv["probability"]=pv["probability"].apply(lambda x:f"{x*100:.1f}%")
pv["gray_score"]=pv["gray_score"].apply(lambda x:f"{x:.3f}")
pv["missing_ratio"]=pv["missing_ratio"].apply(lambda x:f"{x:.0%}")
pv["状态"]=pv["cust_id"].apply(lambda x:"✅" if x in st.session_state.decisions else "▶" if x==cid else "")
pv.columns=["客户号","通过概率","决策区域","灰度分","触发原因","缺失率","状态"]
st.dataframe(pv, use_container_width=True, hide_index=True, height=min(260, 38+len(pv)*35))
