"""auth.py — VitalNav authentication module
SQLite-backed signup / login / logout.

Login modes
-----------
1. Password login  — email + password
2. Face biometric  — webcam snapshot compared to signup photo (>= 70 % similarity)
                     Only the face inside the blue centre-square guide is matched.

Signup captures a webcam selfie (inside a blue square guide) and stores it
as base64 in the DB.  Face Login is only available for accounts with a photo.

All face processing is in face_auth.py (local / CPU-only, no downloads).

Fixes included
--------------
- logout() clears _face_result so Face Login starts clean after logout
- Result card (success / failure) is rendered BEFORE the camera widget so
  Streamlit rerun never wipes the result
- "Try Again" button explicitly clears _face_result and reruns
- Blue square CSS guide injected once at the top of render_auth_page so it
  applies to BOTH signup cam and face-login cam
- crop_center=True passed to compare_faces so only the square region is matched
- No stale .face-match-bar CSS left over
"""
from __future__ import annotations

import base64
import hashlib
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── DB setup ──────────────────────────────────────────────────────────────────

_HERE   = Path(__file__).parent
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
        for col, typedef in [("face_photo", "TEXT"), ("last_login", "TEXT"), ("last_login_face", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass
        conn.commit()


_init_db()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def _photo_to_b64(photo_bytes: bytes) -> str:
    return base64.b64encode(photo_bytes).decode("utf-8")


def _row_to_user(row: sqlite3.Row) -> dict:
    keys  = row.keys()
    photo = row["face_photo"] or (row["last_login_face"] if "last_login_face" in keys else None)
    return {"id": row["id"], "name": row["name"], "email": row["email"], "face_photo": photo}


# ── Core auth functions ───────────────────────────────────────────────────────

def signup(name: str, email: str, password: str) -> tuple[bool, str]:
    name  = name.strip()
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
    """Persist the selfie taken at signup. Returns the base64 string."""
    b64 = _photo_to_b64(photo_bytes)
    with _get_conn() as conn:
        conn.execute("UPDATE users SET face_photo = ? WHERE id = ?", (b64, user_id))
        conn.commit()
    return b64


def login(email: str, password: str) -> tuple[bool, str, dict | None]:
    """Verify credentials, stamp last_login, return user dict."""
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
            return True, f"Welcome back, {row['name']}!", _row_to_user(row)
    return False, "Incorrect email or password. Please try again.", None


def get_all_users_with_photos() -> list[dict]:
    """All users with a stored face photo — used by biometric lookup."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, email, face_photo FROM users "
            "WHERE face_photo IS NOT NULL AND face_photo != ''"
        ).fetchall()
    return [dict(r) for r in rows]


def login_by_user_id(user_id: int) -> dict | None:
    """Stamp last_login and return user dict — called after successful face auth."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now().isoformat(), user_id),
        )
        conn.commit()
        return _row_to_user(row)


# ── Session helpers ───────────────────────────────────────────────────────────

def is_logged_in() -> bool:
    return bool(st.session_state.get("auth_user"))


def current_user() -> dict | None:
    return st.session_state.get("auth_user")


def logout() -> None:
    """
    Clear every auth-related session key so the login page
    always starts completely fresh after logout.
    """
    for key in (
        "auth_user", "auth_tab",
        "_signup_pending_user", "_signup_pending_msg",
        "_face_result",          # ← prevents stale success/fail card on re-login
    ):
        st.session_state.pop(key, None)


# ── Auth wall ─────────────────────────────────────────────────────────────────

def render_auth_page() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, header { visibility: hidden; }

        /* ── All primary buttons ── */
        div[data-testid="stVerticalBlock"] .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #005596 0%, #0077cc 100%);
            color: #fff; border: none; border-radius: 10px;
            padding: 0.65rem 1.2rem; font-weight: 600; font-size: 0.95rem;
            transition: opacity .2s;
        }
        div[data-testid="stVerticalBlock"] .stButton > button:hover { opacity: .88; }

        /* ── Blue square guide overlay on EVERY camera widget ──
           Works for both the signup selfie cam and the face-login cam.       */
        div[data-testid="stCameraInput"] > div {
            position: relative;
        }
        /* Solid blue border square */
        div[data-testid="stCameraInput"] > div::after {
            content: "";
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -54%);
            width: 52%;
            aspect-ratio: 1 / 1;
            border: 3px solid #38bdf8;
            border-radius: 12px;
            /* dark vignette outside the square */
            box-shadow: 0 0 0 3000px rgba(0, 0, 0, 0.28);
            pointer-events: none;
            z-index: 10;
        }
        /* Outer dashed accent ring */
        div[data-testid="stCameraInput"] > div::before {
            content: "";
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -54%);
            width: 52%;
            aspect-ratio: 1 / 1;
            border-radius: 12px;
            outline: 2px dashed rgba(56, 189, 248, 0.40);
            outline-offset: 7px;
            pointer-events: none;
            z-index: 11;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Hero banner ───────────────────────────────────────────────────────────
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
            <div style="color:#e0f2fe;font-size:0.82rem;">&#x2726; AI Symptom Triage</div>
            <div style="color:#e0f2fe;font-size:0.82rem;">&#x2726; Verified Specialists</div>
            <div style="color:#e0f2fe;font-size:0.82rem;">&#x2726; Insurance Plans</div>
            <div style="color:#e0f2fe;font-size:0.82rem;">&#x2726; Cost Estimator</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, card_col, _ = st.columns([1, 2, 1])
    with card_col:

        # ── Signup selfie step (shown right after account creation) ───────────
        if st.session_state.get("_signup_pending_user"):
            _render_signup_photo_step()
            return  # hide the tabs until photo is done

        # ── Three tabs ────────────────────────────────────────────────────────
        tab_pw, tab_face, tab_su = st.tabs(
            ["&#x1F511;  Password Login", "&#x1F464;  Face Login", "&#x2728;  Sign Up"]
        )

        with tab_pw:
            _render_password_login_tab()

        with tab_face:
            _render_face_login_tab()

        with tab_su:
            _render_signup_tab()


# ── Signup photo step ─────────────────────────────────────────────────────────

def _render_signup_photo_step() -> None:
    pending_user = st.session_state["_signup_pending_user"]
    pending_msg  = st.session_state.get("_signup_pending_msg", "")

    st.markdown(
        """
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:14px;
                    padding:1.2rem 1.4rem;margin-bottom:1rem;text-align:center;">
          <div style="font-size:2rem;margin-bottom:0.4rem;">&#x1F4F8;</div>
          <div style="font-weight:700;color:#1e3a5f;font-size:1rem;">One last step — add your photo!</div>
          <div style="font-size:0.82rem;color:#4b6ea8;margin-top:4px;">
            Align your face inside the <strong>blue square</strong>, then click
            <strong>Save &amp; Enter</strong>.<br/>
            This selfie enables Face Login and is stored only on your machine.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align:center;margin-bottom:0.4rem;">
          <div style="background:#0f172a;border-radius:10px;padding:0.45rem 0.8rem;
                      display:inline-block;">
            <span style="color:#93c5fd;font-size:0.78rem;font-weight:600;">
              &#x1F4F7;&nbsp; Centre your face in the blue square
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    photo = st.camera_input("", key="signup_cam", label_visibility="collapsed")

    st.markdown(
        "<p style='text-align:center;font-size:0.75rem;color:#6b7280;margin-top:0.25rem;'>"
        "Good lighting &amp; straight-on angle = best Face Login accuracy</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("&#x1F4F8;  Save & Enter", key="cam_save", disabled=photo is None):
            b64 = save_signup_photo(pending_user["id"], photo.getvalue())
            pending_user["face_photo"] = b64
            st.session_state.auth_user = pending_user
            st.session_state.pop("_signup_pending_user", None)
            st.session_state.pop("_signup_pending_msg", None)
            st.toast(f"Photo saved! Face Login is now active. {pending_msg}")
            st.rerun()
    with c2:
        if st.button("Skip for now", key="cam_skip", type="secondary"):
            st.session_state.auth_user = pending_user
            st.session_state.pop("_signup_pending_user", None)
            st.session_state.pop("_signup_pending_msg", None)
            st.toast(pending_msg + " (No photo — Face Login unavailable.)")
            st.rerun()

    st.markdown(
        "<p style='text-align:center;font-size:0.75rem;color:#9ca3af;margin-top:0.5rem;'>"
        "&#x1F512; Stored securely in your local database only.</p>",
        unsafe_allow_html=True,
    )


# ── Password login tab ────────────────────────────────────────────────────────

def _render_password_login_tab() -> None:
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

    if st.button("Log In  →", key="btn_login"):
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
        "Don't have an account? Switch to the Sign Up tab above.</p>",
        unsafe_allow_html=True,
    )


# ── Face biometric login tab ──────────────────────────────────────────────────

def _render_face_login_tab() -> None:
    try:
        from face_auth import compare_faces, MATCH_THRESHOLD
    except ImportError:
        st.error("face_auth.py not found. Place it in the same folder as auth.py.")
        return

    st.markdown("#### Face Biometric Login")
    st.markdown(
        f"""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                    padding:0.75rem 1rem;margin-bottom:1.1rem;font-size:0.82rem;color:#1e3a5f;">
          &#x1F512; <strong>How it works:</strong> Your live photo is compared to the selfie
          you took at signup. A &ge;&nbsp;{int(MATCH_THRESHOLD * 100)}&nbsp;% facial similarity
          score is required. Only the face <em>inside the blue square</em> is matched —
          no data leaves your machine.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── No face photos registered yet ────────────────────────────────────────
    users_with_photos = get_all_users_with_photos()
    if not users_with_photos:
        st.markdown(
            """
            <div style="background:#fff7ed;border:1.5px solid #fed7aa;border-radius:14px;
                        padding:1.4rem 1.6rem;text-align:center;margin:1rem 0;">
              <div style="font-size:2.2rem;margin-bottom:0.5rem;">&#x1F4ED;</div>
              <div style="font-weight:700;color:#9a3412;font-size:1rem;margin-bottom:0.4rem;">
                No face profiles found in the database
              </div>
              <div style="color:#c2410c;font-size:0.85rem;line-height:1.6;">
                Face Login requires a selfie taken during Sign Up.<br/>
                No account in this database has a face photo yet.
              </div>
              <div style="margin-top:1rem;background:#ffedd5;border-radius:8px;
                          padding:0.65rem 1rem;font-size:0.82rem;color:#7c2d12;">
                &#x1F449; <strong>Sign up</strong> and take a selfie to enable Face Login,
                or use <strong>Password Login</strong> if you already have an account.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Result card — rendered BEFORE the camera so rerun never clears it ────
    #    We store only {"ok": True/False}.  The card stays until:
    #      • "Try Again" is clicked  (failure path)
    #      • logout() is called      (clears _face_result)
    if st.session_state.get("_face_result") is not None:
        res = st.session_state["_face_result"]

        if res["ok"]:
            st.markdown(
                """
                <div style="background:#f0fdf4;border:2px solid #86efac;border-radius:14px;
                            padding:1.8rem 2rem;text-align:center;margin:1rem 0;">
                  <div style="font-size:3rem;margin-bottom:0.5rem;">&#x2705;</div>
                  <div style="font-weight:800;color:#166534;font-size:1.15rem;">Face Verified!</div>
                  <div style="color:#15803d;font-size:0.88rem;margin-top:6px;">Logging you in&#x2026;</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background:#fef2f2;border:2px solid #fca5a5;border-radius:14px;
                            padding:1.8rem 2rem;text-align:center;margin:1rem 0;">
                  <div style="font-size:3rem;margin-bottom:0.5rem;">&#x1F6AB;</div>
                  <div style="font-weight:800;color:#991b1b;font-size:1.15rem;">
                    Not an Authentic User
                  </div>
                  <div style="color:#b91c1c;font-size:0.9rem;margin-top:0.6rem;line-height:1.7;">
                    Your face does not match any registered account in our database.
                  </div>
                  <div style="margin-top:1rem;background:#fee2e2;border-radius:8px;
                              padding:0.65rem 1rem;font-size:0.82rem;color:#7f1d1d;line-height:1.6;">
                    Use <strong>Password Login</strong> or <strong>Sign Up</strong>
                    to create an account.<br/>
                    Tip: good lighting, no glasses, look straight at the camera.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("&#x21A9;  Try Again", key="btn_face_retry", type="secondary"):
                st.session_state.pop("_face_result", None)
                st.rerun()

        # Stop here — never show the camera while a result card is visible
        return

    # ── Camera + Verify button (only when no result is stored) ───────────────
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:0.5rem;">
          <div style="background:#0f172a;border-radius:10px;padding:0.5rem 0.9rem;
                      display:inline-block;">
            <span style="color:#93c5fd;font-size:0.78rem;font-weight:600;">
              &#x1F4F7;&nbsp; Align your face inside the blue square, then click Verify Face
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    live_photo = st.camera_input("", key="face_login_cam", label_visibility="collapsed")

    st.markdown(
        "<p style='text-align:center;font-size:0.75rem;color:#6b7280;margin-top:0.25rem;'>"
        "Centre your face in the blue square &nbsp;&#xB7;&nbsp; good lighting = better accuracy</p>",
        unsafe_allow_html=True,
    )

    if st.button("&#x1F464;  Verify Face", key="btn_face_verify", disabled=live_photo is None):
        live_bytes = live_photo.getvalue()

        with st.spinner("Matching face inside the square&#x2026;"):
            best_score   = 0.0
            best_user    = None
            best_matched = False

            for user_rec in users_with_photos:
                matched, score, _ = compare_faces(
                    user_rec["face_photo"],
                    live_bytes,
                    crop_center=True,   # match ONLY what's inside the blue square
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
            st.session_state["_face_result"] = {"ok": False}
            st.rerun()


# ── Sign-up tab ───────────────────────────────────────────────────────────────

def _render_signup_tab() -> None:
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

    if st.button("Create Account  →", key="btn_signup"):
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