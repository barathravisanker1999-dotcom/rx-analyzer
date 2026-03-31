import os, re, json, torch
import streamlit as st
from PIL import Image
from pathlib import Path
import google.generativeai as genai
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.set_page_config(page_title="Rx Analyzer", page_icon="💊", layout="wide")

st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#0a0e1a}
[data-testid="stSidebar"]{background:#0f1629;border-right:1px solid #1e2d4a}
.rx-header{background:linear-gradient(135deg,#0f1629,#1a2744,#0f3460);padding:2rem 2.5rem;
  border-radius:16px;margin-bottom:1.5rem;border:1px solid #1e3a5f;text-align:center}
.rx-header h1{color:#4fc3f7;font-size:2.5rem;margin:0 0 .3rem 0}
.rx-header p{color:#78909c;margin:0;font-size:.95rem}
.ocr-box{background:#060d1a;border:1px solid #1e3a5f;border-radius:8px;padding:1rem;
  font-family:monospace;font-size:.9rem;color:#81d4fa}
.drug-card{background:#0a1628;border:1px solid #1e3a5f;border-left:4px solid #26c6da;
  border-radius:10px;padding:1rem 1.2rem;margin-bottom:.8rem}
.ai-advice{background:#071a2e;border:1px solid #0d47a1;border-radius:12px;
  padding:1.5rem;color:#e3f2fd;line-height:1.8}
.stButton>button{background:linear-gradient(135deg,#0d47a1,#1565c0);color:white;
  border:none;border-radius:8px;padding:.65rem 1.5rem;font-weight:600}
</style>""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_ID    = "BK1999/rx-trocr-prescription"  # ← CHANGE THIS
IMAGE_SIZE     = (384, 384)
MAX_TEXT_LEN   = 128

# ── Load TrOCR ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading OCR model…")
def load_ocr():
    proc  = TrOCRProcessor.from_pretrained(HF_MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(HF_MODEL_ID).to(DEVICE)
    model.eval()
    return proc, model

# ── Load Gemini ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Gemini…")
def load_gemini():
    key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY","")
    if not key:
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-1.5-flash")

def run_ocr(image, proc, model):
    img = image.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
    pv  = proc(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(pv, max_length=MAX_TEXT_LEN, num_beams=4, early_stopping=True)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

SYSTEM = """You are MedAssist — a compassionate pharmacy assistant AI.
Help patients understand prescriptions and find affordable alternatives.
Never diagnose. Always advise consulting a doctor before any medication change.
Be warm, human and concise."""

def gemini_structure(g, ocr):
    prompt = f"""OCR text from prescription: "{ocr}"
Return ONLY valid JSON (no markdown):
{{"drug_name":"...","brand_name":"...","drug_class":"...","typical_use":"...","common_dosages":["..."],"prescription_needed":true}}
Use null for unknown fields."""
    r = g.generate_content([SYSTEM, prompt],
        generation_config=genai.types.GenerationConfig(temperature=0.1))
    raw = re.sub(r"```json|```","",r.text).strip()
    try:    return json.loads(raw)
    except: return {"drug_name": ocr, "parse_error": True}

def gemini_advice(g, info):
    prompt = f"""Patient prescription: {json.dumps(info,indent=2)}
Write a warm response with emoji headers:
💊 What this medicine does (simple language)
💸 2-3 cheaper generic alternatives
⚠️ Key side effects or food interactions
🌿 Lifestyle tips
💬 Warm encouraging closing
Speak directly to the patient."""
    r = g.generate_content([SYSTEM, prompt],
        generation_config=genai.types.GenerationConfig(temperature=0.7))
    return r.text.strip()

def gemini_qa(g, info, question):
    prompt = f"""Prescription: {json.dumps(info,indent=2)}
Patient asks: "{question}"
Answer warmly and helpfully."""
    r = g.generate_content([SYSTEM, prompt],
        generation_config=genai.types.GenerationConfig(temperature=0.7))
    return r.text.strip()

# ── Session state ───────────────────────────────────────────────────────────
for k,v in {"ocr":None,"info":None,"advice":None,"chat":[]}.items():
    if k not in st.session_state: st.session_state[k] = v

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""<div class="rx-header">
  <h1>💊 Rx Analyzer</h1>
  <p>Handwritten Prescription Intelligence · TrOCR + Gemini 1.5 Flash</p>
</div>""", unsafe_allow_html=True)

proc, ocr_model = load_ocr()
gemini_model    = load_gemini()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Pipeline")
    st.code("Image\n  ↓\nTrOCR (DL)\n  ↓\nGemini 1.5 Flash\n  ↓\nResults + Q&A")
    st.markdown(f"**Gemini:** {'✅ Connected' if gemini_model else '❌ No API key'}")
    st.markdown(f"**Device:** {'GPU' if DEVICE=='cuda' else 'CPU'}")
    st.markdown("---")
    st.markdown("""<div style='background:#1a1200;border:1px solid #f57f17;border-radius:8px;
    padding:.7rem 1rem;color:#ffca28;font-size:.8rem'>
    ⚠️ For informational purposes only. Always consult your doctor before changing medication.
    </div>""", unsafe_allow_html=True)

# ── Upload ──────────────────────────────────────────────────────────────────
st.markdown("### 📸 Upload Prescription")
c1, c2 = st.columns([1,1])
with c1:
    uploaded = st.file_uploader("Upload prescription image",
        type=["jpg","jpeg","png","bmp","tiff"], label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded)
        run_btn = st.button("🔬 Analyze Prescription", use_container_width=True)
with c2:
    if uploaded:
        st.image(image, caption=uploaded.name, use_column_width=True)

# ── Pipeline ─────────────────────────────────────────────────────────────────
if uploaded and run_btn:
    st.markdown("---")
    st.markdown("### 🔡 TrOCR — Text Extraction")
    with st.spinner("Extracting handwritten text…"):
        ocr_text = run_ocr(image, proc, ocr_model)
        st.session_state.ocr = ocr_text
    st.markdown(f"<div class='ocr-box'>Extracted: <b>{ocr_text}</b></div>",
        unsafe_allow_html=True)

    if not ocr_text.strip():
        st.warning("OCR returned empty. Try a clearer image.")
        st.stop()

    st.markdown("---")
    st.markdown("### 🧠 Gemini — Drug Analysis")
    if not gemini_model:
        st.error("Add GEMINI_API_KEY to Streamlit Secrets.")
        st.stop()

    with st.spinner("Identifying drug…"):
        info = gemini_structure(gemini_model, ocr_text)
        st.session_state.info = info

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Drug",  info.get("drug_name")  or "—")
    m2.metric("Brand", info.get("brand_name") or "—")
    m3.metric("Class", info.get("drug_class") or "—")
    rx = info.get("prescription_needed")
    m4.metric("Rx Required", "Yes" if rx else ("No" if rx is False else "—"))

    st.markdown("---")
    st.markdown("### 💬 MedAssist — Advice & Alternatives")
    with st.spinner("Generating advice…"):
        advice = gemini_advice(gemini_model, info)
        st.session_state.advice = advice
    st.markdown(f"<div class='ai-advice'>{advice}</div>", unsafe_allow_html=True)

# ── Q&A Chat ─────────────────────────────────────────────────────────────────
if st.session_state.info:
    st.markdown("---")
    st.markdown("### 💬 Ask MedAssist Anything")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"]=="user" else "💊"):
            st.markdown(msg["content"])
    if q := st.chat_input("Ask about dosage, side effects, alternatives…"):
        st.session_state.chat.append({"role":"user","content":q})
        with st.chat_message("user", avatar="🧑"): st.markdown(q)
        if gemini_model:
            with st.chat_message("assistant", avatar="💊"):
                with st.spinner("Thinking…"):
                    ans = gemini_qa(gemini_model, st.session_state.info, q)
                st.markdown(ans)
            st.session_state.chat.append({"role":"assistant","content":ans})

st.markdown("---")
st.markdown("<center style='color:#37474f;font-size:.78rem'>Rx Analyzer · TrOCR + Gemini 1.5 Flash · Streamlit</center>",
    unsafe_allow_html=True)