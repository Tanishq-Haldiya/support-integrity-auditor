import html
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from sia import PRIORITY_LEVELS, feedback
from sia.audit import audit_frame, needs_attribution
from sia.calibration import urgent_threshold
from sia.explain import occlusion
from sia.models import available_backends, load_model, load_resolution_reference
from sia.rubric import find_cues

st.set_page_config(layout="wide", page_title="Support Integrity Auditor (SIA)")

MISMATCH_COLORS = {"Consistent": "#4c9a6a", "Hidden Crisis": "#d64545", "False Alarm": "#e0a030"}
LEVEL_COLORS = {0: "#dbeafe", 1: "#fde68a", 2: "#fdba74", 3: "#fca5a5"}


@st.cache_resource(show_spinner="Loading model (first start can take ~20s)...")
def get_model(backend):
    return load_model(backend)


@st.cache_resource
def get_reference():
    return load_resolution_reference()


@st.cache_data(show_spinner=False)
def explain(backend, description, subject):
    subject = subject if isinstance(subject, str) else None
    return occlusion(get_model(backend), description, subject, urgent_threshold())


def run_audit(df, backend):
    bar = st.progress(0.0, text="Auditing tickets...")
    result = audit_frame(df, get_model(backend), get_reference(),
                         progress=lambda d, t: bar.progress(d / max(t, 1), text=f"Auditing tickets... {d}/{t}"))
    bar.empty()
    return result


def highlight(text, attributions=()):
    """Escape the ticket text; shade rubric cues by level and underline words the model relied on."""
    spans = [(m.start, m.end, f'<mark style="background:{LEVEL_COLORS[m.level]};padding:0 2px;border-radius:3px" '
                               f'title="rubric cue {m.cue} → {PRIORITY_LEVELS[m.level]}">', "</mark>")
             for m in find_cues(text)]
    spans += [(a["start"], a["end"], f'<span style="border-bottom:3px solid #8b5cf6;font-weight:600" '
                                     f'title="model relied on this (importance {a["importance"]:.2f})">', "</span>")
              for a in attributions]
    out, pos = [], 0
    for start, end, open_tag, close_tag in sorted(spans):
        if start < pos:
            continue
        out.append(html.escape(text[pos:start]))
        out.append(open_tag + html.escape(text[start:end]) + close_tag)
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def show_dossier(row, backend):
    verdict = row["mismatch_type"]
    color = MISMATCH_COLORS[verdict]
    st.markdown(f"<h3 style='color:{color};margin-bottom:0'>{verdict}</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assigned", row["Priority_Level"])
    c2.metric("Inferred", row["inferred_severity"])
    c3.metric("P(High or Critical)", f"{row['p_high_or_critical']:.2f}")
    c4.metric("Confidence", f"{row['confidence']:.2f}")
    if row["needs_review"]:
        st.warning(f"Model and rule-based rubric disagree (rubric says {row['rubric_severity']}). Review manually.")
    dossier = json.loads(row["dossier"]) if isinstance(row.get("dossier"), str) else None
    attributions = []
    if dossier and backend != "rubric" and needs_attribution(dossier):
        # No rubric cue backs this verdict: show which words the model relied on (occlusion).
        with st.spinner("Finding the words the model relied on..."):
            attributions = explain(backend, str(row["Ticket_Description"]), row.get("Ticket_Subject"))
        dossier["feature_evidence"] += attributions
    st.markdown("**Ticket text** (shaded: rubric cues · underlined: words the model relied on)")
    st.markdown(f"<div style='padding:10px;border:1px solid #8884;border-radius:6px'>"
                f"{highlight(str(row['Ticket_Description']), attributions)}</div>", unsafe_allow_html=True)
    if dossier:
        st.markdown("**Constraint analysis**")
        st.write(dossier["constraint_analysis"])
        probs = dossier["model"]["probabilities"]
        fig = px.bar(x=list(probs), y=list(probs.values()), labels={"x": "Severity", "y": "Probability"},
                     range_y=[0, 1], height=250)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Evidence dossier (JSON)"):
            st.json(dossier)
    review_form(row, backend)


def review_form(row, backend):
    """Reviewer decision -> data/feedback.csv, later usable as a training label (train.py --feedback)."""
    with st.form(f"review-{row['Ticket_ID']}"):
        st.markdown("**Reviewer decision**")
        c1, c2 = st.columns([1, 2])
        sev = c1.selectbox("True severity", PRIORITY_LEVELS, index=PRIORITY_LEVELS.index(row["inferred_severity"]))
        note = c2.text_input("Note (optional)")
        if st.form_submit_button("Save review"):
            feedback.record(row.to_dict(), backend, row["mismatch_type"], row["inferred_severity"], sev, note)
            st.success(f"Saved. {len(feedback.load())} ticket(s) reviewed so far.")


# --- Sidebar -------------------------------------------------------------------
backends = available_backends()
with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Severity model", backends,
                           help="hybrid = rubric where it has a problem cue, the learned model elsewhere. "
                                "embedding, distilbert and tfidf are trained by train.py; "
                                "rubric is the rule-based labeller and needs no training.")
    if backends == ["rubric"]:
        st.info("No trained models found. Run `python train.py` to add tfidf/distilbert.")
    st.caption(f"A ticket counts as urgent when P(High or Critical) ≥ {urgent_threshold():.2f}. "
               "**Hidden Crisis**: urgent ticket labelled Low/Medium. "
               "**False Alarm**: routine ticket labelled High/Critical.")

# Load (and cache) the model up front so the first "Analyze" click is instant.
get_model(backend)

st.title("Support Integrity Auditor (SIA)")
st.write("Detects priority mismatch in customer support tickets and explains each flag with evidence "
         "taken directly from the ticket.")

tabs = st.tabs(["Single ticket", "Batch audit", "Dashboard"])

with tabs[0]:
    with st.form("single_ticket_form"):
        col1, col2 = st.columns(2)
        with col1:
            ticket_id = st.text_input("Ticket ID", "TKT-TEST")
            subject = st.text_input("Subject", "Application crashing")
        with col2:
            priority = st.selectbox("Assigned priority", PRIORITY_LEVELS)
            res_time = st.number_input("Resolution time (hours, optional)", value=24.0, min_value=0.0)
        description = st.text_area("Description", "The application crashes every time I open the settings tab.")
        submit = st.form_submit_button("Analyze ticket")
    if submit:
        one = pd.DataFrame([{"Ticket_ID": ticket_id, "Ticket_Subject": subject, "Priority_Level": priority,
                             "Resolution_Time_Hours": res_time, "Ticket_Description": description}])
        # Kept in session state so the result (and its review form) survive the rerun that
        # any later button click triggers.
        st.session_state["single"] = (audit_frame(one, get_model(backend), get_reference()).iloc[0], backend)
    if "single" in st.session_state:
        show_dossier(*st.session_state["single"])

with tabs[1]:
    st.write("Upload a CSV with at least `Ticket_ID`, `Ticket_Description` and `Priority_Level` "
             "(`Ticket_Subject` and `Resolution_Time_Hours` are optional).")
    uploaded = st.file_uploader("Tickets CSV", type=["csv"])
    c1, c2 = st.columns([1, 3])
    use_sample = c1.button("Use bundled dataset")
    sample_n = c2.slider("Rows from bundled dataset", 500, 20000, 5000, step=500)
    source = None
    if uploaded is not None and st.button("Audit uploaded file", type="primary"):
        source = pd.read_csv(uploaded)
    elif use_sample:
        source = pd.read_csv("data/customer_support_tickets.csv").head(sample_n)
    if source is not None:
        try:
            st.session_state["results"] = run_audit(source, backend)
            st.session_state["results_backend"] = backend
        except ValueError as e:
            st.error(str(e))
    if "results" in st.session_state:
        res = st.session_state["results"]
        counts = res["mismatch_type"].value_counts()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tickets", len(res))
        m2.metric("Hidden Crisis", int(counts.get("Hidden Crisis", 0)))
        m3.metric("False Alarm", int(counts.get("False Alarm", 0)))
        m4.metric("Needs review", int(res["needs_review"].sum()))
        cols = ["Ticket_ID", "Priority_Level", "inferred_severity", "mismatch_type", "confidence",
                "needs_review", "Ticket_Description"]
        st.dataframe(res[cols].head(200), width="stretch", hide_index=True)
        st.download_button("Download results CSV", res.to_csv(index=False).encode(),
                           file_name="sia_batch_results.csv", mime="text/csv")

with tabs[2]:
    if "results" not in st.session_state:
        st.info("Run a batch audit first (Batch audit tab).")
    else:
        df = st.session_state["results"]
        st.caption(f"{len(df)} tickets audited with the {st.session_state['results_backend']} backend.")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df, names="mismatch_type", title="Mismatch type distribution",
                         color="mismatch_type", color_discrete_map=MISMATCH_COLORS)
            st.plotly_chart(fig, width="stretch")
        with col2:
            heat = pd.crosstab(df["Priority_Level"], df["inferred_severity"]).reindex(
                index=PRIORITY_LEVELS, columns=PRIORITY_LEVELS, fill_value=0)
            fig = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                            labels={"x": "Inferred severity", "y": "Assigned priority"},
                            title="Assigned priority vs inferred severity")
            st.plotly_chart(fig, width="stretch")

        flagged = df[df["mismatch_type"] != "Consistent"]
        group_col = "Issue_Category" if "Issue_Category" in df else ("Ticket_Channel" if "Ticket_Channel" in df else None)
        col3, col4 = st.columns(2)
        if group_col and len(flagged):
            with col3:
                fig = px.histogram(flagged, x=group_col, color="mismatch_type", barmode="group",
                                   color_discrete_map=MISMATCH_COLORS, title=f"Mismatches by {group_col}")
                st.plotly_chart(fig, width="stretch")
        if len(flagged):
            with col4:
                fig = px.histogram(flagged, x="confidence", nbins=20, color="mismatch_type",
                                   color_discrete_map=MISMATCH_COLORS, title="Confidence of flagged tickets")
                st.plotly_chart(fig, width="stretch")

        stats = feedback.agreement(feedback.load())
        if stats.get("flags_reviewed"):
            st.info(f"Reviewers upheld {stats['flags_upheld']} of {stats['flags_reviewed']} reviewed flags "
                    f"(precision {stats['flag_precision']:.0%}). Retrain with "
                    "`python train.py --feedback` to learn from them.")

        st.subheader("Review queue")
        f1, f2, f3 = st.columns(3)
        types = f1.multiselect("Mismatch type", ["Hidden Crisis", "False Alarm"], ["Hidden Crisis", "False Alarm"])
        min_conf = f2.slider("Minimum confidence", 0.5, 1.0, 0.5, 0.01)
        only_review = f3.checkbox("Only model/rubric disagreements")
        queue = flagged[flagged["mismatch_type"].isin(types) & (flagged["confidence"] >= min_conf)]
        if only_review:
            queue = queue[queue["needs_review"]]
        # Hidden crises first: under-prioritised urgent tickets are the costlier error.
        queue = queue.sort_values(["mismatch_type", "confidence"], ascending=[False, False])
        st.dataframe(queue[["Ticket_ID", "Priority_Level", "inferred_severity", "mismatch_type", "confidence",
                            "needs_review", "Ticket_Description"]], width="stretch", hide_index=True)
        if len(queue):
            pick = st.selectbox("Inspect ticket", queue["Ticket_ID"].tolist())
            show_dossier(queue[queue["Ticket_ID"] == pick].iloc[0], st.session_state["results_backend"])
