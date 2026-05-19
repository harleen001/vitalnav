"""VitalNav - Jalandhar's AI-powered health navigator
Streamlit + Groq (llama-3.3-70b-versatile). Production-grade healthcare dashboard
inspired by Bajaj Finserv Health.

Auth: SQLite-backed signup/login (auth.py). DB stored at /tmp/vitalnav_users.db.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# ---------------------------------------------------------------------------
# Page config  (must be FIRST st call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VitalNav · Jalandhar",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Auth — imported AFTER set_page_config
# ---------------------------------------------------------------------------
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))  # ensure auth.py is found next to app.py
from auth import is_logged_in, current_user, logout, render_auth_page  # noqa: E402

# Gate: show auth wall if not logged in
if not is_logged_in():
    render_auth_page()
    st.stop()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


inject_custom_css()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "data" / "jalandhar_data.json"
    with open(data_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    doctors = pd.DataFrame(data["doctors"])
    hospitals = pd.DataFrame(data["hospitals"])
    plans = pd.DataFrame(data["plans"])
    procedures = pd.DataFrame(data.get("procedures", []))
    return data["city"], doctors, hospitals, plans, procedures


CITY, DOCTORS_DF, HOSPITALS_DF, PLANS_DF, PROCEDURES_DF = load_data()
HOSPITAL_BY_ID = {h["id"]: h for h in HOSPITALS_DF.to_dict("records")}
DOCTOR_BY_ID = {d["id"]: d for d in DOCTORS_DF.to_dict("records")}
PLAN_BY_ID = {p["id"]: p for p in PLANS_DF.to_dict("records")}


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
@st.cache_resource
def get_groq_client():
    key = st.secrets.get("GROQ_API_KEY")
    if not key:
        key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)


GROQ_MODEL = "llama-3.3-70b-versatile"

DIAGNOSIS_SYSTEM_PROMPT = """You are a careful medical triage assistant for VitalNav,
serving patients in Jalandhar, Punjab, India. Given a patient's symptom description
and optional health profile context, return STRICTLY a JSON object (no preamble, no
markdown fences) of this exact shape:

{
  "conditions": [
    {
      "name": "<short condition name>",
      "confidence": <integer 0-100>,
      "description": "<one sentence plain-language description>",
      "specialty": "<one of: Cardiologist, Pulmonologist, Gastroenterologist, Neurologist, Neurosurgeon, Orthopedist, Dermatologist, Endocrinologist, ENT Specialist, Psychiatrist, General Physician>",
      "urgency": "<one of: Low, Moderate, High, Emergency>"
    }
  ],
  "summary": "<1-2 sentence empathetic observation, mentioning the patient's profile if relevant>",
  "red_flags": ["<symptom or sign that warrants urgent attention, if any>"]
}

Rules:
- Return EXACTLY 3 conditions, ordered by confidence descending.
- Confidence reflects realistic likelihood, not severity.
- Use the patient's health profile (age, allergies, medications, conditions) to refine
  your differential when provided.
- Use common, recognisable condition names that match a typical health database
  (e.g. "Migraine", "Asthma", "GERD", "Type 2 Diabetes", "Hypertension", "Jaundice",
  "Hepatitis", "Common Cold", "Flu", "Anxiety", "Eczema", "Arthritis", "Sinusitis",
  "Stroke", "Brain Tumor", "Spine Disorder").
- "red_flags" should be empty list [] unless symptoms genuinely warrant urgent care.
- Never recommend medications. This is triage, not diagnosis.
- Output VALID JSON only.
"""


def build_profile_context(profile: dict) -> str:
    if not profile or not any(profile.values()):
        return ""
    parts = []
    if profile.get("age"):
        parts.append(f"Age: {profile['age']}")
    if profile.get("sex") and profile["sex"] != "Prefer not to say":
        parts.append(f"Sex: {profile['sex']}")
    if profile.get("conditions"):
        parts.append(f"Existing conditions: {profile['conditions']}")
    if profile.get("medications"):
        parts.append(f"Current medications: {profile['medications']}")
    if profile.get("allergies"):
        parts.append(f"Known allergies: {profile['allergies']}")
    return "Patient health profile — " + " | ".join(parts) if parts else ""


def groq_query(user_input: str, profile: dict | None = None):
    client = get_groq_client()
    if client is None:
        return None, "AI service is not configured. Please add your GROQ_API_KEY."
    profile_ctx = build_profile_context(profile or {})
    user_content = f"{profile_ctx}\n\nSymptom report: {user_input}" if profile_ctx else user_input
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if "conditions" not in parsed or not isinstance(parsed.get("conditions"), list):
            return None, "AI returned an unexpected response. Please try again."
        return parsed, None
    except json.JSONDecodeError:
        return None, "AI returned an unparseable response. Please try rephrasing."
    except Exception as exc:  # noqa: BLE001
        return None, f"AI service error: {exc}"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "chat_history": [],
        "diagnosis": None,
        "selected_condition": None,
        "active_phase": "Symptom Check",
        "booked_doctors": set(),
        "subscribed_plans": set(),
        "compare_plans": set(),
        "assessment_history": [],
        "hospital_filter": "All hospitals",
        "health_profile": {
            "age": None,
            "sex": "Prefer not to say",
            "conditions": "",
            "medications": "",
            "allergies": "",
        },
        "show_profile": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def initials(name: str) -> str:
    parts = [p for p in name.replace("Dr.", "").strip().split() if p]
    if not parts:
        return "DR"
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()


def cond_match(condition: str, lst: list[str]) -> bool:
    cl = condition.lower()
    return any(cl in c.lower() or c.lower() in cl for c in lst)


def doctors_for_condition(condition: str, specialty: str | None = None,
                          hospital_id: str | None = None) -> pd.DataFrame:
    matches = DOCTORS_DF[DOCTORS_DF["conditions"].apply(lambda lst: cond_match(condition, lst))]
    if specialty and not matches.empty:
        spec_match = matches[matches["specialty"] == specialty]
        if not spec_match.empty:
            matches = spec_match
    if matches.empty and specialty:
        matches = DOCTORS_DF[DOCTORS_DF["specialty"] == specialty]
    if hospital_id and hospital_id != "all":
        h_match = matches[matches["hospital_id"] == hospital_id]
        if not h_match.empty:
            matches = h_match
    return matches.sort_values(by=["rating", "experience_years"], ascending=False)


def plans_for_condition(condition: str, hospital_id: str | None = None) -> pd.DataFrame:
    cl = condition.lower()
    cond_keywords = {cl}

    def covers(row):
        text = " ".join(row.get("covered", [])).lower() + " " + row.get("best_for", "").lower()
        return any(k in text for k in cond_keywords)
    matches = PLANS_DF[PLANS_DF.apply(covers, axis=1)]
    if matches.empty:
        matches = PLANS_DF[PLANS_DF["category"].isin(["Family Floater", "Comprehensive", "Premium Family Floater"])]
    if hospital_id and hospital_id != "all":
        hosp_name = HOSPITAL_BY_ID.get(hospital_id, {}).get("name", "")
        if hosp_name:
            in_network = matches[matches["cashless_jalandhar"].apply(lambda lst: hosp_name in lst)]
            if not in_network.empty:
                matches = in_network
    return matches.sort_values(by="price_monthly")


def urgency_color(level: str) -> str:
    return {
        "Low": "#10b981",
        "Moderate": "#f59e0b",
        "High": "#ef4444",
        "Emergency": "#dc2626",
    }.get(level, "#6b7280")


# ---------------------------------------------------------------------------
# Header  (now shows logged-in user + logout)
# ---------------------------------------------------------------------------
def profile_summary_chip() -> str:
    p = st.session_state.health_profile
    bits = []
    if p.get("age"):
        bits.append(f"{p['age']} yrs")
    if p.get("sex") and p["sex"] != "Prefer not to say":
        bits.append(p["sex"])
    if p.get("conditions"):
        bits.append("conditions ✓")
    if p.get("medications"):
        bits.append("meds ✓")
    if p.get("allergies"):
        bits.append("allergies ✓")
    return " · ".join(bits) if bits else "Not set up"


user = current_user()
hdr_left, hdr_right = st.columns([5, 2])
with hdr_left:
    st.markdown(
        f"""
        <div class="vn-header">
          <div class="vn-logo">
            <div class="vn-logo-mark">V+</div>
            <div class="vn-logo-text">
              <div class="name">VitalNav</div>
              <div class="tag">Your AI health navigator · serving {CITY['name']}, {CITY['state']}</div>
            </div>
          </div>
          <div class="vn-header-right">
            <div class="vn-loc-pill">
              <span class="dot"></span> {CITY['name']}, {CITY['state']}
            </div>
            <a class="vn-emergency" href="tel:{CITY['emergency_numbers']['ambulance']}">
              Emergency · {CITY['emergency_numbers']['ambulance']}
            </a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hdr_right:
    # Build face avatar or initials fallback
    face_b64 = user.get("face_photo") or ""
    if face_b64:
        avatar_html = (
            f'''<img src="data:image/jpeg;base64,{face_b64}"
                 style="width:46px;height:46px;border-radius:50%;
                        object-fit:cover;border:2px solid #005596;
                        vertical-align:middle;margin-right:10px;">''')
    else:
        ini = (user["name"][0] if user.get("name") else "U").upper()
        avatar_html = (
            f'''<div style="width:46px;height:46px;border-radius:50%;
                          background:linear-gradient(135deg,#005596,#0077cc);
                          display:inline-flex;align-items:center;justify-content:center;
                          color:#fff;font-weight:800;font-size:1.1rem;
                          vertical-align:middle;margin-right:10px;">{ini}</div>''')

    st.markdown(
        f"""<div class="vn-profile-pill" style="display:flex;align-items:center;">
              {avatar_html}
              <div>
                <div class="vn-profile-pill-label" style="font-weight:700;">{user['name']}</div>
                <div class="vn-profile-pill-value">{user['email']}</div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Edit profile", key="toggle_profile", type="secondary"):
            st.session_state.show_profile = not st.session_state.show_profile
            st.rerun()
    with btn_col2:
        if st.button("Log Out", key="btn_logout", type="secondary"):
            logout()
            st.rerun()

if st.session_state.show_profile:
    with st.container():
        st.markdown('<div class="vn-profile-panel">', unsafe_allow_html=True)
        st.markdown(
            "#### Personal health profile\n"
            "These details are kept in your session and sent to the AI as context for "
            "more accurate triage. Leave anything blank if you'd rather not share."
        )
        p = st.session_state.health_profile
        c1, c2 = st.columns(2)
        with c1:
            new_age = st.number_input("Age", min_value=0, max_value=120, value=p.get("age") or 0, step=1)
            new_sex = st.selectbox(
                "Sex",
                ["Prefer not to say", "Female", "Male", "Other"],
                index=["Prefer not to say", "Female", "Male", "Other"].index(p.get("sex", "Prefer not to say")),
            )
        with c2:
            new_conditions = st.text_input(
                "Existing conditions",
                value=p.get("conditions", ""),
                placeholder="e.g. Hypertension, Type 2 Diabetes",
            )
            new_meds = st.text_input(
                "Current medications",
                value=p.get("medications", ""),
                placeholder="e.g. Metformin 500mg, Atorvastatin",
            )
        new_allergies = st.text_input(
            "Known allergies",
            value=p.get("allergies", ""),
            placeholder="e.g. Penicillin, peanuts, pollen",
        )
        sb1, sb2, _ = st.columns([1, 1, 4])
        with sb1:
            if st.button("Save profile", key="save_profile"):
                st.session_state.health_profile = {
                    "age": int(new_age) if new_age else None,
                    "sex": new_sex,
                    "conditions": new_conditions.strip(),
                    "medications": new_meds.strip(),
                    "allergies": new_allergies.strip(),
                }
                st.session_state.show_profile = False
                st.toast("Health profile saved — future assessments will use this context.")
                st.rerun()
        with sb2:
            if st.button("Clear", key="clear_profile", type="secondary"):
                st.session_state.health_profile = {
                    "age": None, "sex": "Prefer not to say", "conditions": "",
                    "medications": "", "allergies": "",
                }
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Phase nav
# ---------------------------------------------------------------------------
phases = ["Symptom Check", "Find Doctors", "Hospitals", "Health Plans", "Cost Estimator", "My Dashboard"]
phase_idx = phases.index(st.session_state.active_phase) if st.session_state.active_phase in phases else 0
st.session_state.active_phase = st.radio(
    "Phase",
    phases,
    index=phase_idx,
    horizontal=True,
    label_visibility="collapsed",
    key="phase_radio",
)


# ---------------------------------------------------------------------------
# Diagnosis renderer
# ---------------------------------------------------------------------------
def render_diagnosis_block(diag: dict) -> None:
    if not diag or "conditions" not in diag:
        return
    if diag.get("summary"):
        st.markdown(
            f"""<div class="vn-banner">
                  <div>
                    <div class="title">AI Triage Summary</div>
                    <div class="sub">{diag['summary']}</div>
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )
    if diag.get("red_flags"):
        rf = "<br/>".join(f"⚠ {f}" for f in diag["red_flags"])
        st.markdown(
            f"""<div class="vn-redflag">
                  <strong>Red flags noticed:</strong><br/>{rf}<br/>
                  <small>If any of these are severe or worsening, call 108 or visit the nearest emergency room.</small>
                </div>""",
            unsafe_allow_html=True,
        )
    for cond in diag["conditions"]:
        urg = cond.get("urgency", "Moderate")
        urg_col = urgency_color(urg)
        st.markdown(
            f"""
            <div class="vn-diag" style="border-left-color: {urg_col};">
              <div class="vn-diag-head">
                <p class="vn-diag-name">{cond.get('name', '—')}</p>
                <span class="vn-diag-conf">{cond.get('confidence', 0)}% match</span>
              </div>
              <p class="vn-diag-desc">{cond.get('description', '')}</p>
              <div class="vn-diag-spec">
                Recommended: <strong>{cond.get('specialty', 'General Physician')}</strong>
                <span class="vn-urg-pill" style="background:{urg_col}1a;color:{urg_col};">{urg} urgency</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    chart_df = pd.DataFrame(diag["conditions"])
    fig = go.Figure(go.Bar(
        x=chart_df["confidence"], y=chart_df["name"], orientation="h",
        marker=dict(color="#005596"),
        text=[f"{c}%" for c in chart_df["confidence"]], textposition="outside",
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=dict(text="Confidence breakdown", font=dict(size=14, color="#1a1a1a")),
        xaxis=dict(range=[0, 110], showgrid=True, gridcolor="#eee", title=""),
        yaxis=dict(autorange="reversed", title=""),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=20, t=40, b=10), height=220,
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """<div class="vn-disclaimer">
            <strong>Disclaimer:</strong> AI triage is not a medical diagnosis.
            Always consult a qualified doctor for clinical decisions.
            In an emergency, call 108 immediately.
           </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Doctor card
# ---------------------------------------------------------------------------
def render_doctor_card(row: pd.Series, idx: int) -> None:
    booked = row["id"] in st.session_state.booked_doctors
    langs = ", ".join(row["languages"][:3])
    quals = row.get("qualifications", "")
    st.markdown(
        f"""
        <div class="vn-card vn-doctor">
          <div class="vn-doctor-head">
            <div class="vn-doctor-avatar">{initials(row['name'])}</div>
            <div style="flex:1;min-width:0;">
              <p class="vn-doctor-name">{row['name']}</p>
              <p class="vn-doctor-quals">{quals}</p>
              <p class="vn-doctor-specialty">{row['specialty']} · {row['experience_years']} yrs</p>
            </div>
          </div>
          <div class="vn-doctor-meta">
            <span class="vn-chip">★ {row['rating']} ({row.get('reviews', 0)})</span>
            <span class="vn-chip vn-chip-orange">{row['specialty']}</span>
          </div>
          <div class="vn-doctor-info">
            <div>📍 {row['hospital']}</div>
            <div>🗣 {langs}</div>
            <div>🕐 Next: <strong>{row.get('next_slot', 'Call to confirm')}</strong></div>
          </div>
          <div class="vn-doctor-fee">
            ₹{row['consultation_fee']:,} <small>/ consultation</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    label = "✓ Appointment booked" if booked else "Book Appointment"
    if st.button(label, key=f"book_{row['id']}_{idx}", disabled=booked):
        st.session_state.booked_doctors.add(row["id"])
        st.toast(f"Appointment requested with {row['name']}")
        st.rerun()


# ---------------------------------------------------------------------------
# Hospital card
# ---------------------------------------------------------------------------
def render_hospital_card(row: pd.Series) -> None:
    specs = " · ".join(row["specialties"][:4])
    insurers = ", ".join(row["cashless_insurers"][:4])
    if len(row["cashless_insurers"]) > 4:
        insurers += f" +{len(row['cashless_insurers']) - 4} more"
    highlights_html = "".join(f"<li>{h}</li>" for h in row["highlights"])
    st.markdown(
        f"""
        <div class="vn-card vn-hospital">
          <div class="vn-hospital-head">
            <div>
              <h4 class="vn-hospital-name">{row['name']}</h4>
              <p class="vn-hospital-tag">{row['tagline']}</p>
            </div>
            <div class="vn-hospital-rating">
              <div class="stars">★ {row['rating']}</div>
              <div class="reviews">{row['reviews']:,} reviews</div>
            </div>
          </div>
          <div class="vn-hospital-stats">
            <div><span class="num">{row['beds']}</span><span class="lbl">beds</span></div>
            <div><span class="num">{2026 - row['established']}+</span><span class="lbl">years</span></div>
            <div><span class="num">{len(row['specialties'])}</span><span class="lbl">specialties</span></div>
          </div>
          <div class="vn-hospital-info">
            <div>📍 {row['address']}</div>
            <div>📞 {row['phone']}</div>
          </div>
          <div class="vn-hospital-section">
            <div class="lbl">Top specialties</div>
            <div class="val">{specs}</div>
          </div>
          <div class="vn-hospital-section">
            <div class="lbl">Cashless insurers</div>
            <div class="val">{insurers}</div>
          </div>
          <ul class="vn-plan-list">{highlights_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Plan card
# ---------------------------------------------------------------------------
def render_plan_card(row: pd.Series, idx: int) -> None:
    subscribed = row["id"] in st.session_state.subscribed_plans
    covered_html = "".join(f"<li>{c}</li>" for c in row["covered"])
    not_covered_html = "".join(f"<li>{c}</li>" for c in row["not_covered"])
    cashless = ", ".join(row["cashless_jalandhar"][:3])
    if len(row["cashless_jalandhar"]) > 3:
        cashless += f" +{len(row['cashless_jalandhar']) - 3} more"
    si_lakhs = row["sum_insured"] / 100000
    st.markdown(
        f"""
        <div class="vn-card vn-plan">
          <div class="vn-plan-head">
            <span class="vn-plan-cat">{row['category']}</span>
            <span class="vn-plan-provider">{row['provider']}</span>
          </div>
          <h4 class="vn-plan-name">{row['name']}</h4>
          <div class="vn-plan-price">
            <span class="amount">₹{row['price_monthly']:,}</span>
            <span class="period">/month · ₹{row['price_annual']:,}/yr</span>
          </div>
          <div class="vn-plan-coverage">
            Sum insured up to <span>₹{si_lakhs:.0f} L</span>
          </div>
          <div class="vn-plan-section">
            <div class="lbl ok">✓ What's covered</div>
            <ul class="vn-plan-list">{covered_html}</ul>
          </div>
          <div class="vn-plan-section">
            <div class="lbl no">✗ What's not covered</div>
            <ul class="vn-plan-list muted">{not_covered_html}</ul>
          </div>
          <div class="vn-plan-section">
            <div class="lbl">🏥 Cashless in Jalandhar</div>
            <div class="val">{cashless}</div>
          </div>
          <div class="vn-plan-best-for">Best for: {row['best_for']}</div>
          <div class="vn-plan-uin">IRDAI UIN: {row.get('irda_uin', '—')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    label = "✓ Subscribed" if subscribed else "Get This Plan"
    if st.button(label, key=f"plan_{row['id']}_{idx}", disabled=subscribed):
        st.session_state.subscribed_plans.add(row["id"])
        st.toast(f"{row['name']} added — our advisor will call shortly.")
        st.rerun()


# ---------------------------------------------------------------------------
# PHASE 1 — Symptom Check
# ---------------------------------------------------------------------------
def phase_symptom_check() -> None:
    st.markdown('<h2 class="vn-section-title">Tell us how you\'re feeling</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="vn-section-sub">Describe your symptoms and our AI will identify likely conditions and connect you to the right specialist in {CITY["name"]}.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Verified Doctors", f"{len(DOCTORS_DF)}", f"Across {CITY['name']}"),
        ("Network Hospitals", f"{len(HOSPITALS_DF)}", "Cashless ready"),
        ("Insurance Plans", f"{len(PLANS_DF)}", "From 8 IRDAI insurers"),
        ("Avg AI Response", "< 2 sec", "Powered by Groq"),
    ]
    for col, (label, value, delta) in zip([c1, c2, c3, c4], stats):
        col.markdown(
            f"""<div class="vn-stat">
                  <div class="label">{label}</div>
                  <div class="value">{value}</div>
                  <div class="delta">{delta}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    chat_col, side_col = st.columns([3, 2], gap="large")

    with chat_col:
        st.markdown("#### AI Health Assistant")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else "🧑"):
                st.markdown(msg["content"])

        if not st.session_state.chat_history:
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown(
                    f"Namaste — I'm your VitalNav assistant. Describe what you're "
                    f"experiencing — symptoms, when they started, anything unusual — "
                    f"and I'll identify the most likely conditions and suggest a "
                    f"specialist in {CITY['name']}."
                )
            st.markdown("**Or try a quick example:**")
            qcols = st.columns(3)
            quick_prompts = [
                "Sharp chest pain when I climb stairs, getting worse over a week",
                "Persistent dry cough and shortness of breath at night",
                "Yellow eyes, dark urine, no appetite for last 3 days",
            ]
            for col, prompt in zip(qcols, quick_prompts):
                if col.button(prompt, key=f"quick_{prompt[:12]}"):
                    handle_user_message(prompt)
                    st.rerun()

        user_msg = st.chat_input("Describe your symptoms…")
        if user_msg:
            handle_user_message(user_msg)
            st.rerun()

    with side_col:
        st.markdown("#### Latest assessment")
        if st.session_state.diagnosis:
            render_diagnosis_block(st.session_state.diagnosis)
            if st.button("→ See specialists in Jalandhar", key="goto_doctors", type="secondary"):
                st.session_state.active_phase = "Find Doctors"
                st.rerun()
        else:
            st.markdown(
                """<div class="vn-empty">
                     <div class="icon">🩺</div>
                     <div class="title">No assessment yet</div>
                     <div>Share your symptoms in the chat to get an AI-powered health analysis.</div>
                   </div>""",
                unsafe_allow_html=True,
            )


def handle_user_message(text: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": text})
    with st.spinner("Analyzing your symptoms…"):
        diag, err = groq_query(text, st.session_state.health_profile)
    if err or not diag:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": err or "I couldn't process that. Could you add more detail?"}
        )
        return
    st.session_state.diagnosis = diag
    if diag.get("conditions"):
        st.session_state.selected_condition = diag["conditions"][0]["name"]
    st.session_state.assessment_history.insert(0, {
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "input": text,
        "diagnosis": diag,
    })
    st.session_state.assessment_history = st.session_state.assessment_history[:20]
    bullet_lines = "\n".join(
        f"- **{c.get('name', '—')}** — {c.get('confidence', 0)}% match · "
        f"see a {c.get('specialty', 'General Physician')} "
        f"({c.get('urgency', 'Moderate')} urgency)"
        for c in diag.get("conditions", [])
    )
    profile_note = ""
    if any(st.session_state.health_profile.get(k) for k in ("age", "conditions", "medications", "allergies")):
        profile_note = "\n\n_Used your saved health profile for context._"
    rf_note = ""
    if diag.get("red_flags"):
        rf_note = "\n\n⚠ **Red flags noted** — see the assessment panel."
    reply = (
        f"{diag.get('summary', 'Here is what I picked up from your description.')}\n\n"
        f"{bullet_lines}\n\n"
        "Open **Find Doctors** to book a consultation. *Disclaimer: this is not a medical diagnosis.*"
        f"{rf_note}{profile_note}"
    )
    st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# PHASE 2 — Find Doctors
# ---------------------------------------------------------------------------
def phase_find_doctors() -> None:
    st.markdown(f'<h2 class="vn-section-title">Specialists in {CITY["name"]}</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="vn-section-sub">Verified doctors with real qualifications, ratings and next available slots.</p>',
        unsafe_allow_html=True,
    )

    fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
    with fcol1:
        if st.session_state.diagnosis and st.session_state.diagnosis.get("conditions"):
            condition_names = [c["name"] for c in st.session_state.diagnosis["conditions"]]
            default_idx = (condition_names.index(st.session_state.selected_condition)
                           if st.session_state.selected_condition in condition_names else 0)
            st.session_state.selected_condition = st.selectbox(
                "Showing for condition",
                condition_names,
                index=default_idx,
            )
        else:
            st.markdown("&nbsp;")
    with fcol2:
        specialty_filter = st.selectbox(
            "Specialty",
            ["All"] + sorted(DOCTORS_DF["specialty"].unique().tolist()),
        )
    with fcol3:
        hospital_options = {"All hospitals": "all"} | {h["name"]: h["id"] for h in HOSPITALS_DF.to_dict("records")}
        hospital_label = st.selectbox("Hospital", list(hospital_options.keys()))
        hospital_id = hospital_options[hospital_label]

    spec_filter = None if specialty_filter == "All" else specialty_filter

    if st.session_state.diagnosis and st.session_state.diagnosis.get("conditions"):
        cond = st.session_state.selected_condition
        spec_for_cond = next(
            (c.get("specialty") for c in st.session_state.diagnosis["conditions"] if c["name"] == cond),
            None,
        )
        results = doctors_for_condition(cond, spec_filter or spec_for_cond, hospital_id)
    else:
        results = DOCTORS_DF.copy()
        if spec_filter:
            results = results[results["specialty"] == spec_filter]
        if hospital_id != "all":
            results = results[results["hospital_id"] == hospital_id]
        results = results.sort_values(by=["rating", "experience_years"], ascending=False)

    st.markdown(f"<div class='vn-result-count'>{len(results)} doctors match your filters</div>",
                unsafe_allow_html=True)

    if results.empty:
        st.markdown(
            """<div class="vn-empty">
                 <div class="icon">😕</div>
                 <div class="title">No matches</div>
                 <div>Try clearing the hospital or specialty filter.</div>
               </div>""",
            unsafe_allow_html=True,
        )
        return

    rows = results.to_dict("records")
    for i in range(0, len(rows), 3):
        cols = st.columns(3, gap="medium")
        for col, row in zip(cols, rows[i : i + 3]):
            with col:
                render_doctor_card(pd.Series(row), i)

    if st.session_state.booked_doctors:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.success(
            f"You've booked {len(st.session_state.booked_doctors)} consultation(s). "
            f"Our team will call within 30 minutes to confirm timing."
        )


# ---------------------------------------------------------------------------
# PHASE 3 — Hospitals
# ---------------------------------------------------------------------------
def phase_hospitals() -> None:
    st.markdown(f'<h2 class="vn-section-title">Top hospitals in {CITY["name"]}</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="vn-section-sub">Multi-specialty care providers across Jalandhar, with bed counts, specialties and cashless insurance partners.</p>',
        unsafe_allow_html=True,
    )

    fcol1, fcol2 = st.columns([2, 4])
    with fcol1:
        all_specs = sorted({s for sp in HOSPITALS_DF["specialties"] for s in sp})
        spec = st.selectbox("Filter by specialty", ["All"] + all_specs)

    results = HOSPITALS_DF.copy()
    if spec != "All":
        results = results[results["specialties"].apply(lambda s: spec in s)]
    results = results.sort_values(by=["rating", "reviews"], ascending=False)

    st.markdown(f"<div class='vn-result-count'>{len(results)} hospitals</div>", unsafe_allow_html=True)

    rows = results.to_dict("records")
    for i in range(0, len(rows), 2):
        cols = st.columns(2, gap="medium")
        for col, row in zip(cols, rows[i : i + 2]):
            with col:
                render_hospital_card(pd.Series(row))


# ---------------------------------------------------------------------------
# PHASE 4 — Health Plans
# ---------------------------------------------------------------------------
def phase_health_plans() -> None:
    st.markdown('<h2 class="vn-section-title">Health insurance plans</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="vn-section-sub">IRDAI-regulated plans from leading Indian insurers, with cashless coverage at Jalandhar hospitals.</p>',
        unsafe_allow_html=True,
    )

    fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
    with fcol1:
        if st.session_state.diagnosis and st.session_state.diagnosis.get("conditions"):
            condition_names = [c["name"] for c in st.session_state.diagnosis["conditions"]]
            chosen_cond = st.selectbox("Plans relevant to", condition_names, index=0)
        else:
            chosen_cond = None
            st.markdown("&nbsp;")
    with fcol2:
        provider = st.selectbox("Insurer", ["All"] + sorted(PLANS_DF["provider"].unique().tolist()))
    with fcol3:
        hospital_options = {"All hospitals": "all"} | {h["name"]: h["id"] for h in HOSPITALS_DF.to_dict("records")}
        hospital_label = st.selectbox("Cashless at", list(hospital_options.keys()), key="plan_hosp")
        hospital_id = hospital_options[hospital_label]

    if chosen_cond:
        results = plans_for_condition(chosen_cond, hospital_id if hospital_id != "all" else None)
    else:
        results = PLANS_DF.copy()
        if hospital_id != "all":
            hosp_name = HOSPITAL_BY_ID.get(hospital_id, {}).get("name", "")
            if hosp_name:
                results = results[results["cashless_jalandhar"].apply(lambda lst: hosp_name in lst)]

    if provider != "All":
        results = results[results["provider"] == provider]
    results = results.sort_values(by="price_monthly")

    st.markdown(f"<div class='vn-result-count'>{len(results)} plans available</div>",
                unsafe_allow_html=True)

    if results.empty:
        st.markdown(
            """<div class="vn-empty">
                 <div class="icon">📋</div>
                 <div class="title">No plans match these filters</div>
                 <div>Try changing the insurer or hospital.</div>
               </div>""",
            unsafe_allow_html=True,
        )
        return

    cmp_count = len(st.session_state.compare_plans)
    cmp_col1, cmp_col2 = st.columns([4, 1])
    with cmp_col1:
        st.markdown(
            f"<div class='vn-compare-bar'>"
            f"<strong>{cmp_count}</strong> selected for comparison "
            f"<span class='vn-compare-hint'>· tick the box on any plan to compare up to 3 side-by-side</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with cmp_col2:
        if cmp_count >= 2 and st.button(f"Compare {cmp_count} plans", key="open_compare"):
            st.session_state["_show_compare"] = True

    if st.session_state.get("_show_compare") and cmp_count >= 2:
        render_compare_table()
        if st.button("Close comparison", key="close_compare", type="secondary"):
            st.session_state["_show_compare"] = False
            st.rerun()

    rows = results.to_dict("records")
    for i in range(0, len(rows), 3):
        cols = st.columns(3, gap="medium")
        for col, row in zip(cols, rows[i : i + 3]):
            with col:
                render_plan_card(pd.Series(row), i)
                in_compare = row["id"] in st.session_state.compare_plans
                cmp_label = "✓ In comparison" if in_compare else "+ Add to compare"
                disabled = (not in_compare) and len(st.session_state.compare_plans) >= 3
                if st.button(cmp_label, key=f"cmp_{row['id']}_{i}", type="secondary",
                             disabled=disabled and not in_compare):
                    if in_compare:
                        st.session_state.compare_plans.discard(row["id"])
                    else:
                        st.session_state.compare_plans.add(row["id"])
                    st.rerun()

    if st.session_state.subscribed_plans:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.success(
            f"{len(st.session_state.subscribed_plans)} plan(s) added. "
            "An IRDAI-licensed advisor will reach out for KYC and policy issuance."
        )


def render_compare_table() -> None:
    selected = [PLAN_BY_ID[pid] for pid in st.session_state.compare_plans if pid in PLAN_BY_ID]
    if not selected:
        return
    headers = "".join(
        f"<th><div class='cmp-prov'>{p['provider']}</div>"
        f"<div class='cmp-name'>{p['name']}</div></th>"
        for p in selected
    )

    def row(label, fn):
        cells = "".join(f"<td>{fn(p)}</td>" for p in selected)
        return f"<tr><th class='cmp-lbl'>{label}</th>{cells}</tr>"

    def covered_cell(p):
        return "<ul class='cmp-list'>" + "".join(f"<li>✓ {c}</li>" for c in p["covered"]) + "</ul>"

    def not_covered_cell(p):
        return "<ul class='cmp-list cmp-no'>" + "".join(f"<li>× {c}</li>" for c in p["not_covered"]) + "</ul>"

    def cashless_cell(p):
        return "<br/>".join(p["cashless_jalandhar"])

    body = (
        row("Category", lambda p: p["category"]) +
        row("Sum insured", lambda p: f"₹{p['sum_insured']/100000:.0f} L") +
        row("Monthly premium", lambda p: f"₹{p['price_monthly']:,}") +
        row("Annual premium", lambda p: f"₹{p['price_annual']:,}") +
        row("IRDAI UIN", lambda p: p.get("irda_uin", "—")) +
        row("Best for", lambda p: p["best_for"]) +
        row("What's covered", covered_cell) +
        row("What's NOT covered", not_covered_cell) +
        row("Cashless in Jalandhar", cashless_cell)
    )
    st.markdown(
        f"""<div class='vn-compare-wrap'>
              <table class='vn-compare-table'>
                <thead><tr><th class='cmp-lbl'>Feature</th>{headers}</tr></thead>
                <tbody>{body}</tbody>
              </table>
            </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PHASE 5 — Cost Estimator
# ---------------------------------------------------------------------------
def phase_cost_estimator() -> None:
    st.markdown('<h2 class="vn-section-title">Procedure Cost Estimator</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="vn-section-sub">Indicative price ranges for common procedures across {CITY["name"]} hospitals. '
        f'Final cost depends on implant choice, room category, and complications.</p>',
        unsafe_allow_html=True,
    )

    if PROCEDURES_DF.empty:
        st.info("No procedure data available.")
        return

    fcol1, _ = st.columns([2, 4])
    with fcol1:
        cats = ["All"] + sorted(PROCEDURES_DF["category"].unique().tolist())
        cat = st.selectbox("Category", cats)

    procs = PROCEDURES_DF.copy()
    if cat != "All":
        procs = procs[procs["category"] == cat]

    rows = procs.to_dict("records")
    for i in range(0, len(rows), 2):
        cols = st.columns(2, gap="medium")
        for col, p in zip(cols, rows[i : i + 2]):
            with col:
                hospitals_list = "".join(
                    f"<li>{HOSPITAL_BY_ID.get(hid, {}).get('name', hid)}</li>"
                    for hid in p.get("preferred_hospitals", [])
                )
                covered_html = (
                    "<span class='vn-cost-cov ok'>✓ Typically covered by health insurance</span>"
                    if p.get("covered_by_typical_plan")
                    else "<span class='vn-cost-cov no'>✗ Usually NOT covered (out-of-pocket)</span>"
                )
                col.markdown(
                    f"""
                    <div class="vn-card vn-cost-card">
                      <div class="vn-cost-cat">{p['category']}</div>
                      <h4 class="vn-cost-name">{p['name']}</h4>
                      <div class="vn-cost-range">
                        <span class="lo">₹{p['price_min']:,}</span>
                        <span class="dash">→</span>
                        <span class="hi">₹{p['price_max']:,}</span>
                      </div>
                      <div class="vn-cost-stay">🏨 Hospital stay: <strong>{p['stay_days']}</strong></div>
                      <div class="vn-cost-section">
                        <div class="lbl">Recommended hospitals in {CITY['name']}</div>
                        <ul class="vn-plan-list">{hospitals_list}</ul>
                      </div>
                      {covered_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# PHASE 6 — My Dashboard
# ---------------------------------------------------------------------------
def phase_dashboard() -> None:
    st.markdown('<h2 class="vn-section-title">My Health Dashboard</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="vn-section-sub">Everything you\'ve done in this session — assessments, appointments, plans.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    p = st.session_state.health_profile
    profile_done = sum(1 for v in (p.get("age"), p.get("conditions"), p.get("medications"), p.get("allergies")) if v)
    stats = [
        ("Assessments", str(len(st.session_state.assessment_history)), "AI triages this session"),
        ("Appointments", str(len(st.session_state.booked_doctors)), "Booked"),
        ("Active Plans", str(len(st.session_state.subscribed_plans)), "Subscribed"),
        ("Profile completeness", f"{profile_done}/4", "Health fields filled"),
    ]
    for col, (label, value, delta) in zip([c1, c2, c3, c4], stats):
        col.markdown(
            f"""<div class="vn-stat">
                  <div class="label">{label}</div>
                  <div class="value">{value}</div>
                  <div class="delta">{delta}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("#### Your appointments")
        if not st.session_state.booked_doctors:
            st.markdown(
                """<div class="vn-empty">
                     <div class="icon">📅</div>
                     <div class="title">No appointments yet</div>
                     <div>Book a doctor from the Find Doctors phase to see them here.</div>
                   </div>""",
                unsafe_allow_html=True,
            )
        else:
            for did in list(st.session_state.booked_doctors):
                d = DOCTOR_BY_ID.get(did)
                if not d:
                    continue
                st.markdown(
                    f"""
                    <div class="vn-card vn-dash-row">
                      <div class="vn-dash-avatar">{initials(d['name'])}</div>
                      <div class="vn-dash-body">
                        <div class="vn-dash-title">{d['name']}</div>
                        <div class="vn-dash-sub">{d['specialty']} · {d['hospital']}</div>
                        <div class="vn-dash-meta">🕐 {d.get('next_slot', '—')} · ₹{d['consultation_fee']:,}</div>
                      </div>
                      <div class="vn-dash-tag ok">Confirmed</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Cancel", key=f"cancel_appt_{did}", type="secondary"):
                    st.session_state.booked_doctors.discard(did)
                    st.rerun()

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("#### Assessment history")
        if not st.session_state.assessment_history:
            st.markdown(
                """<div class="vn-empty">
                     <div class="icon">📝</div>
                     <div class="title">No history yet</div>
                     <div>Run a symptom check to start building your health timeline.</div>
                   </div>""",
                unsafe_allow_html=True,
            )
        else:
            for idx, a in enumerate(st.session_state.assessment_history):
                top = (a["diagnosis"].get("conditions") or [{}])[0]
                cond_name = top.get("name", "—")
                conf = top.get("confidence", 0)
                with st.expander(f"{a['timestamp']} · Top: {cond_name} ({conf}% match)"):
                    st.markdown(f"**Your input:** _{a['input']}_")
                    st.markdown(f"**Summary:** {a['diagnosis'].get('summary', '—')}")
                    for c in a["diagnosis"].get("conditions", []):
                        st.markdown(
                            f"- **{c.get('name')}** — {c.get('confidence', 0)}% · "
                            f"{c.get('specialty', '')} · "
                            f"<span style='color:{urgency_color(c.get('urgency','Moderate'))};font-weight:700'>"
                            f"{c.get('urgency')}</span>",
                            unsafe_allow_html=True,
                        )

    with right:
        st.markdown("#### Active plans")
        if not st.session_state.subscribed_plans:
            st.markdown(
                """<div class="vn-empty">
                     <div class="icon">💼</div>
                     <div class="title">No plans yet</div>
                     <div>Browse Health Plans to get covered.</div>
                   </div>""",
                unsafe_allow_html=True,
            )
        else:
            for pid in list(st.session_state.subscribed_plans):
                pl = PLAN_BY_ID.get(pid)
                if not pl:
                    continue
                st.markdown(
                    f"""
                    <div class="vn-card vn-dash-plan">
                      <div class="vn-dash-plan-prov">{pl['provider']}</div>
                      <div class="vn-dash-plan-name">{pl['name']}</div>
                      <div class="vn-dash-plan-meta">
                        ₹{pl['price_monthly']:,}/mo · ₹{pl['sum_insured']/100000:.0f} L cover
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Remove", key=f"rm_plan_{pid}", type="secondary"):
                    st.session_state.subscribed_plans.discard(pid)
                    st.rerun()

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("#### Health profile snapshot")
        prof = st.session_state.health_profile
        rows_html = ""
        for label, val in [
            ("Age", prof.get("age") or "—"),
            ("Sex", prof.get("sex") if prof.get("sex") != "Prefer not to say" else "—"),
            ("Conditions", prof.get("conditions") or "—"),
            ("Medications", prof.get("medications") or "—"),
            ("Allergies", prof.get("allergies") or "—"),
        ]:
            rows_html += f"<div class='vn-prof-row'><span class='lbl'>{label}</span><span class='val'>{val}</span></div>"
        st.markdown(f"<div class='vn-card'>{rows_html}</div>", unsafe_allow_html=True)

        # ── Logged-in account info ──────────────────────────────────────────
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("#### Account")
        u = current_user()
        face_b64 = u.get("face_photo") or ""
        if face_b64:
            face_img = (
                f'''<div style="text-align:center;margin-bottom:1rem;">
                  <img src="data:image/jpeg;base64,{face_b64}"
                       style="width:88px;height:88px;border-radius:50%;
                              object-fit:cover;border:3px solid #005596;
                              box-shadow:0 4px 16px rgba(0,85,150,0.18);">
                  <div style="font-size:0.75rem;color:#6b7280;margin-top:6px;">
                    📸 Last login photo
                  </div>
                </div>''')
        else:
            ini = (u["name"][0] if u.get("name") else "U").upper()
            face_img = (
                f'''<div style="text-align:center;margin-bottom:1rem;">
                  <div style="width:88px;height:88px;border-radius:50%;
                              background:linear-gradient(135deg,#005596,#0077cc);
                              display:inline-flex;align-items:center;justify-content:center;
                              color:#fff;font-weight:800;font-size:2rem;">{ini}</div>
                  <div style="font-size:0.75rem;color:#9ca3af;margin-top:6px;">
                    No photo yet — taken on next login
                  </div>
                </div>''')
        st.markdown(
            f"""<div class='vn-card'>
                  {face_img}
                  <div class='vn-prof-row'>
                    <span class='lbl'>Name</span><span class='val'>{u['name']}</span>
                  </div>
                  <div class='vn-prof-row'>
                    <span class='lbl'>Email</span><span class='val'>{u['email']}</span>
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Log Out", key="dashboard_logout", type="secondary"):
            logout()
            st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
def render_footer() -> None:
    st.markdown(
        f"""
        <div class="vn-footer">
          <div class="vn-footer-grid">
            <div>
              <div class="vn-footer-brand">VitalNav</div>
              <div class="vn-footer-tag">{CITY['name']}'s AI health navigator</div>
              <div class="vn-footer-small">
                Verified hospital data sourced from public listings.<br/>
                IRDAI-regulated insurance plans only.
              </div>
            </div>
            <div>
              <div class="vn-footer-h">For Patients</div>
              <a>AI Symptom Check</a><a>Find a Doctor</a><a>Hospital Network</a><a>Health Plans</a>
            </div>
            <div>
              <div class="vn-footer-h">Emergency</div>
              <a href="tel:{CITY['emergency_numbers']['ambulance']}">Ambulance · {CITY['emergency_numbers']['ambulance']}</a>
              <a href="tel:{CITY['emergency_numbers']['police']}">Police · {CITY['emergency_numbers']['police']}</a>
              <a href="tel:{CITY['emergency_numbers']['civil_hospital']}">Civil Hospital, {CITY['name']}</a>
            </div>
            <div>
              <div class="vn-footer-h">Legal</div>
              <a>Disclaimer</a><a>Privacy</a><a>Terms</a><a>IRDAI Compliance</a>
            </div>
          </div>
          <div class="vn-footer-bottom">
            © 2026 VitalNav · This platform is for triage and information only and does not substitute professional medical advice.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
phase = st.session_state.active_phase
if phase == "Symptom Check":
    phase_symptom_check()
elif phase == "Find Doctors":
    phase_find_doctors()
elif phase == "Hospitals":
    phase_hospitals()
elif phase == "Health Plans":
    phase_health_plans()
elif phase == "Cost Estimator":
    phase_cost_estimator()
elif phase == "My Dashboard":
    phase_dashboard()

render_footer()