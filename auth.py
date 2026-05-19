"""auth.py — VitalNav authentication module
SQLite-backed signup / login / logout.

Login modes
-----------
1. Password login  — email + password (unchanged)
2. Face biometric  — webcam snapshot compared to signup photo (≥ 70 % similarity)

Signup still captures a webcam selfie and stores it as base64 in the DB.
"""
import base64
import hashlib
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st

# ─── DB setup ───────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
DB_PATH = str(_HERE / "vitalnav_users.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                email       TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                password    TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                face_photo  TEXT,
                last_login  TEXT
            )
            """
        )
        for col, typedef in [("face_photo", "TEXT"), ("last_login", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN last_login_face TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()


_init_db()


# ─── Helpers ────────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def _photo_to_b64(photo_bytes: bytes) -> str:
    return base64.b64encode(photo_bytes).decode("utf-8")


# ─── Core auth functions ─────────────────────────────────────────────────────

def signup(name: str, email: str, password: str) -> tuple[bool, str]:
    name = name.strip()
    email = email.strip().lower()
    if not name or len(name) < 2:
        return False, "Please enter your full name (at least 2 characters)."
    if not _valid_email(email):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (name, email, password, created_at) VALUES (?, ?, ?, ?)",
                (name, email, _hash(password), datetime.now().isoformat()),
            )
            conn.commit()
        return True, f"Account created! Welcome, {name}."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists. Please log in."


def save_signup_photo(user_id: int, photo_bytes: bytes) -> str:
    """Save the face photo captured at signup. Returns the base64 string."""
    b64 = _photo_to_b64(photo_bytes)
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET face_photo = ? WHERE id = ?",
            (b64, user_id),
        )
        conn.commit()
    return b64


def login(email: str, password: str) -> tuple[bool, str, dict | None]:
    """Verify credentials, record last_login timestamp, return user dict with face_photo."""
    email = email.strip().lower()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? AND password = ?",
            (email, _hash(password)),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now().isoformat(), row["id"]),
            )
            conn.commit()
            photo = row["face_photo"] or (
                row["last_login_face"] if "last_login_face" in row.keys() else None
            )
            user = {
                "id":         row["id"],
                "name":       row["name"],
                "email":      row["email"],
                "face_photo": photo,
            }
            return True, f"Welcome back, {row['name']}!", user
    return False, "Incorrect email or password. Please try again.", None


def get_all_users_with_photos() -> list[dict]:
    """Return all users that have a face_photo stored (for biometric lookup)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, email, face_photo FROM users WHERE face_photo IS NOT NULL AND face_photo != ''"
        ).fetchall()
    return [dict(r) for r in rows]


def login_by_user_id(user_id: int) -> dict | None:
    """Stamp last_login and return user dict for a known user_id (used after face auth)."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now().isoformat(), user_id),
        )
        conn.commit()
        photo = row["face_photo"] or (
            row["last_login_face"] if "last_login_face" in row.keys() else None
        )
        return {
            "id":         row["id"],
            "name":       row["name"],
            "email":      row["email"],
            "face_photo": photo,
        }


# ─── Session helpers ─────────────────────────────────────────────────────────

def is_logged_in() -> bool:
    return bool(st.session_state.get("auth_user"))


def current_user() -> dict | None:
    return st.session_state.get("auth_user")


def logout() -> None:
    for key in ("auth_user", "auth_tab", "_signup_pending_user", "_signup_pending_msg",
                "_face_result", "_face_last_img_id"):
        st.session_state.pop(key, None)


# ─── Auth wall UI ─────────────────────────────────────────────────────────────

def render_auth_page() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, header { visibility: hidden; }
        .cam-prompt {
            background: #eff6ff; border: 1px solid #bfdbfe;
            border-radius: 14px; padding: 1.2rem 1.4rem;
            margin-bottom: 1rem; text-align: center;
        }
        .cam-prompt .icon  { font-size: 2rem; margin-bottom: 0.4rem; }
        .cam-prompt .title { font-weight: 700; color: #1e3a5f; font-size: 1rem; }
        .cam-prompt .sub   { font-size: 0.82rem; color: #4b6ea8; margin-top: 4px; }
        div[data-testid="stVerticalBlock"] .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #005596 0%, #0077cc 100%);
            color: #fff; border: none; border-radius: 10px;
            padding: 0.65rem 1.2rem; font-weight: 600; font-size: 0.95rem;
            transition: opacity .2s;
        }
        div[data-testid="stVerticalBlock"] .stButton > button:hover { opacity: .88; }
        .face-match-bar {
            background: #f0fdf4; border: 1px solid #bbf7d0;
            border-radius: 10px; padding: 0.9rem 1.1rem;
            margin: 0.6rem 0; font-size: 0.9rem;
        }
        .face-match-bar.fail {
            background: #fef2f2; border-color: #fecaca;
        }
        .face-score-track {
            background: #e5e7eb; border-radius: 9999px;
            height: 8px; margin-top: 6px; overflow: hidden;
        }
        .face-score-fill {
            height: 8px; border-radius: 9999px;
            transition: width 0.4s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(120deg,#005596 0%,#0099e6 100%);
                    padding:2.8rem 2rem 2rem; text-align:center;
                    border-radius:0 0 32px 32px; margin-bottom:2rem;">
          <div style="font-size:2rem;font-weight:900;color:#fff;letter-spacing:-1px;">VitalNav</div>
          <div style="color:#bde0ff;font-size:0.95rem;margin-top:6px;">
            Jalandhar's AI-powered health navigator
          </div>
          <div style="display:flex;gap:1.5rem;justify-content:center;margin-top:1.4rem;flex-wrap:wrap;">
            <div style="color:#e0f2fe;font-size:0.82rem;">✦ AI Symptom Triage</div>
            <div style="color:#e0f2fe;font-size:0.82rem;">✦ Verified Specialists</div>
            <div style="color:#e0f2fe;font-size:0.82rem;">✦ Insurance Plans</div>
            <div style="color:#e0f2fe;font-size:0.82rem;">✦ Cost Estimator</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, card_col, _ = st.columns([1, 2, 1])
    with card_col:

        # ── WEBCAM STEP: only for signup ──────────────────────────────────
        if st.session_state.get("_signup_pending_user"):
            pending_user = st.session_state["_signup_pending_user"]
            pending_msg  = st.session_state.get("_signup_pending_msg", "")

            st.markdown(
                """
                <div class="cam-prompt">
                  <div class="icon">📸</div>
                  <div class="title">One last step — add your photo!</div>
                  <div class="sub">
                    Take a quick selfie to personalise your account and enable
                    <strong>Face Login</strong>.<br/>
                    This photo is stored securely in your local database.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            photo = st.camera_input("", key="signup_cam", label_visibility="collapsed")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("📸  Save & Enter", key="cam_save", disabled=photo is None):
                    b64 = save_signup_photo(pending_user["id"], photo.getvalue())
                    pending_user["face_photo"] = b64
                    st.session_state.auth_user = pending_user
                    st.session_state.pop("_signup_pending_user", None)
                    st.session_state.pop("_signup_pending_msg", None)
                    st.toast(f"📸 Photo saved! Face Login is now enabled. {pending_msg}")
                    st.rerun()
            with c2:
                if st.button("Skip →", key="cam_skip", type="secondary"):
                    st.session_state.auth_user = pending_user
                    st.session_state.pop("_signup_pending_user", None)
                    st.session_state.pop("_signup_pending_msg", None)
                    st.toast(pending_msg + " (No photo — Face Login unavailable.)")
                    st.rerun()

            st.markdown(
                "<p style='text-align:center;font-size:0.75rem;color:#9ca3af;margin-top:0.6rem;'>"
                "🔒 Stored securely in your local database only.</p>",
                unsafe_allow_html=True,
            )
            return  # hide tabs while on webcam step

        # ── LOGIN / SIGNUP TABS ──────────────────────────────────────────
        tab_login, tab_face, tab_signup = st.tabs(
            [" Password Login", " Face Login", " Sign Up"]
        )

        # ── TAB 1: PASSWORD LOGIN ─────────────────────────────────────────
        with tab_login:
            st.markdown("#### Welcome back")
            st.markdown(
                "<p style='color:#6b7280;font-size:0.88rem;margin-bottom:1.2rem;'>"
                "Log in with your email and password.</p>",
                unsafe_allow_html=True,
            )
            login_email = st.text_input("Email address", key="li_email", placeholder="you@example.com")
            login_pass  = st.text_input("Password", type="password", key="li_pass", placeholder="••••••••")

            if "li_error" in st.session_state:
                st.error(st.session_state.pop("li_error"))

            if st.button("Log In →", key="btn_login"):
                ok, msg, user = login(login_email, login_pass)
                if ok:
                    st.session_state.auth_user = user
                    st.toast(msg)
                    st.rerun()
                else:
                    st.session_state.li_error = msg
                    st.rerun()

            st.markdown(
                "<p style='text-align:center;font-size:0.8rem;color:#9ca3af;margin-top:1rem;'>"
                "Don't have an account? Switch to the Sign Up tab ↑</p>",
                unsafe_allow_html=True,
            )

        # ── TAB 2: FACE BIOMETRIC LOGIN ───────────────────────────────────
        with tab_face:
            _render_face_login_tab()

        # ── TAB 3: SIGNUP ─────────────────────────────────────────────────
        with tab_signup:
            st.markdown("#### Create your account")
            st.markdown(
                "<p style='color:#6b7280;font-size:0.88rem;margin-bottom:1.2rem;'>"
                "Free forever. No credit card required.</p>",
                unsafe_allow_html=True,
            )
            su_name  = st.text_input("Full name",        key="su_name",  placeholder="Kartik Attri")
            su_email = st.text_input("Email address",    key="su_email", placeholder="username@gmail.com")
            su_pass  = st.text_input("Password",         type="password", key="su_pass",  placeholder="Min. 6 characters")
            su_pass2 = st.text_input("Confirm password", type="password", key="su_pass2", placeholder="Repeat password")

            if "su_error" in st.session_state:
                st.error(st.session_state.pop("su_error"))

            if st.button("Create Account →", key="btn_signup"):
                if su_pass != su_pass2:
                    st.session_state.su_error = "Passwords do not match."
                    st.rerun()
                else:
                    ok, msg = signup(su_name, su_email, su_pass)
                    if ok:
                        _, _, user = login(su_email, su_pass)
                        st.session_state["_signup_pending_user"] = user
                        st.session_state["_signup_pending_msg"]  = msg
                        st.rerun()
                    else:
                        st.session_state.su_error = msg
                        st.rerun()

            st.markdown(
                "<p style='text-align:center;font-size:0.78rem;color:#9ca3af;margin-top:1rem;'>"
                "By signing up you agree to our Terms of Use.<br/>"
                "This platform is for triage only — not a substitute for medical advice.</p>",
                unsafe_allow_html=True,
            )


# ── Face Login sub-UI ─────────────────────────────────────────────────────────

def _render_face_login_tab() -> None:
    """Render the biometric face-login UI."""
    # Lazy import so face_auth is optional
    try:
        from face_auth import compare_faces, has_face, MATCH_THRESHOLD
    except ImportError:
        st.error("face_auth.py not found — place it in the same directory as auth.py.")
        return

    st.markdown("#### Face Biometric Login")
    st.markdown(
        "<p style='color:#6b7280;font-size:0.88rem;margin-bottom:0.4rem;'>"
        "Look directly at the camera, then click <strong>Verify Face</strong>.</p>",
        unsafe_allow_html=True,
    )

    # Info box
    st.markdown(
        f"""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                    padding:0.75rem 1rem;margin-bottom:1rem;font-size:0.82rem;color:#1e3a5f;">
          🔒 <strong>How it works:</strong> Your live photo is compared to the selfie you
          took at signup. A ≥&nbsp;{int(MATCH_THRESHOLD*100)}&nbsp;% facial similarity score
          is required to log you in. No data leaves your machine.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check there are any users with photos
    users_with_photos = get_all_users_with_photos()
    if not users_with_photos:
        st.markdown(
            """
            <div style="
                background: #fff7ed;
                border: 1.5px solid #fed7aa;
                border-radius: 14px;
                padding: 1.4rem 1.6rem;
                text-align: center;
                margin: 1rem 0;
            ">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📭</div>
                <div style="font-weight: 700; color: #9a3412; font-size: 1rem; margin-bottom: 0.4rem;">
                    No face profiles found in the database
                </div>
                <div style="color: #c2410c; font-size: 0.85rem; line-height: 1.6;">
                    Face Login requires a selfie taken at signup.<br/>
                    No account in this database has a face photo yet.
                </div>
                <div style="
                    margin-top: 1rem;
                    background: #ffedd5;
                    border-radius: 8px;
                    padding: 0.65rem 1rem;
                    font-size: 0.82rem;
                    color: #7c2d12;
                ">
                    👉 <strong>Sign up</strong> and take a selfie to enable Face Login, or use
                    <strong>Password Login</strong> if you already have an account.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # If we already have a result stored, show ONLY the result — no camera
    if st.session_state.get("_face_result") is not None:
        res = st.session_state["_face_result"]

        if res["ok"]:
            st.markdown(
                """
                <div style="background:#f0fdf4;border:2px solid #86efac;border-radius:14px;
                            padding:1.6rem 1.8rem;text-align:center;margin:1rem 0;">
                  <div style="font-size:2.8rem;margin-bottom:0.5rem;">&#x2705;</div>
                  <div style="font-weight:800;color:#166534;font-size:1.1rem;">Face Verified!</div>
                  <div style="color:#15803d;font-size:0.88rem;margin-top:6px;">Logging you in...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background:#fef2f2;border:2px solid #fca5a5;border-radius:14px;
                            padding:1.6rem 1.8rem;text-align:center;margin:1rem 0;">
                  <div style="font-size:2.8rem;margin-bottom:0.5rem;">&#x1F6AB;</div>
                  <div style="font-weight:800;color:#991b1b;font-size:1.15rem;letter-spacing:-0.2px;">
                    Not an Authentic User
                  </div>
                  <div style="color:#b91c1c;font-size:0.9rem;margin-top:0.6rem;line-height:1.7;">
                    Your face does not match any registered account in our database.
                  </div>
                  <div style="margin-top:1rem;background:#fee2e2;border-radius:8px;
                              padding:0.6rem 1rem;font-size:0.82rem;color:#7f1d1d;line-height:1.6;">
                    Please use <strong>Password Login</strong> or <strong>Sign Up</strong> to create an account.<br/>
                    Tip: remove glasses, improve lighting, and look straight at the camera.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Try Again", key="btn_face_retry", type="secondary"):
                st.session_state.pop("_face_result", None)
                st.rerun()
        return  # don't render camera again until they hit Try Again

    # No result yet — show camera + verify button
    live_photo = st.camera_input(
        "Position your face in the centre of the frame",
        key="face_login_cam",
    )

    if st.button(
        "👤  Verify Face",
        key="btn_face_verify",
        disabled=live_photo is None,
    ):
        live_bytes = live_photo.getvalue()

        with st.spinner("Scanning your face against registered accounts…"):
            best_score   = 0.0
            best_user    = None
            best_matched = False

            for user_rec in users_with_photos:
                matched, score, _ = compare_faces(
                    user_rec["face_photo"], live_bytes
                )
                if score > best_score:
                    best_score   = score
                    best_user    = user_rec
                    best_matched = matched

        if best_matched and best_user:
            user = login_by_user_id(best_user["id"])
            if user:
                st.session_state["_face_result"] = {"ok": True}
                st.session_state.auth_user = user
                st.toast(f"Face verified! Welcome back, {user['name']}.")
                st.rerun()
        else:
            # Face not recognised — store failure and rerun to show message
            st.session_state["_face_result"] = {"ok": False}
            st.rerun()

    st.markdown(
        "<p style='text-align:center;font-size:0.78rem;color:#9ca3af;margin-top:0.8rem;'>"
        "Ensure good lighting and look straight at the camera.</p>",
        unsafe_allow_html=True,
    )