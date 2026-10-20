import streamlit as st
from PIL import Image
from transformers import pipeline
from deepface import DeepFace
import tempfile
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Multimodal AI Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PREMIUM DARK UI
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #090d18;
    --panel: #11182a;
    --border: rgba(148, 163, 184, 0.16);
    --text: #edf1ff;
    --muted: #9aa7c7;
    --purple: #9b8cff;
    --blue: #64a8ff;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(108, 76, 255, 0.15), transparent 34%),
        radial-gradient(ellipse at 90% 8%, rgba(48, 132, 255, 0.12), transparent 30%),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.hero {
    position: relative;
    padding: 34px;
    border-radius: 25px;
    border: 1px solid rgba(155, 140, 255, 0.25);
    background: linear-gradient(
        125deg,
        rgba(112, 82, 255, 0.16),
        rgba(28, 52, 95, 0.28),
        rgba(17, 24, 42, 0.90)
    );
    margin-bottom: 25px;
}

.eyebrow {
    color: #b6aaff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2.2px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(34px, 5vw, 54px);
    font-weight: 800;
    letter-spacing: -2px;
    margin: 12px 0 8px;
    color: #f5f6ff;
}

.gradient-text {
    background: linear-gradient(90deg, #b4a4ff, #6ebaff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #b4bfd9;
    max-width: 780px;
    font-size: 15px;
    line-height: 1.8;
}

.badge {
    display: inline-block;
    border: 1px solid rgba(111, 231, 190, 0.25);
    color: #80e8bd;
    background: rgba(54, 211, 153, 0.07);
    border-radius: 50px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 700;
    margin-top: 8px;
}

.section-heading {
    font-size: 23px;
    font-weight: 750;
    color: #f1f3ff;
    margin: 22px 0 5px;
}

.section-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 20px;
}

.panel {
    background: linear-gradient(
        145deg,
        rgba(20, 29, 49, 0.95),
        rgba(13, 19, 34, 0.95)
    );
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 23px;
    min-height: 185px;
}

.panel-title {
    color: #f0f2ff;
    font-size: 18px;
    font-weight: 750;
    margin-bottom: 5px;
}

.panel-description {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.7;
    margin-bottom: 17px;
}

.result-card {
    border: 1px solid rgba(151, 135, 255, 0.25);
    background: linear-gradient(
        120deg,
        rgba(105, 78, 255, 0.12),
        rgba(45, 100, 180, 0.09)
    );
    border-radius: 17px;
    padding: 22px;
    margin: 12px 0;
}

.result-label {
    color: #a9b6d4;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 1.3px;
    font-weight: 750;
}

.result-value {
    color: #c2b6ff;
    font-size: 29px;
    font-weight: 800;
    margin-top: 7px;
    overflow-wrap: anywhere;
}

.small-note {
    color: #96a3c2;
    font-size: 12px;
    line-height: 1.8;
}

div.stButton > button {
    width: 100%;
    border: 1px solid rgba(174, 157, 255, 0.45);
    border-radius: 12px;
    min-height: 48px;
    font-weight: 750;
    color: white;
    background: linear-gradient(95deg, #7055e9, #377fe8);
    transition: 0.2s ease;
}

div.stButton > button:hover {
    border-color: #c1b4ff;
    box-shadow: 0 0 25px rgba(112, 85, 233, 0.22);
    transform: translateY(-1px);
    color: white;
}

.stTextArea textarea {
    background: #0b1222 !important;
    color: #eef2ff !important;
    border: 1px solid #283550 !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploader"] {
    background: rgba(9, 15, 29, 0.55);
    border: 1px dashed #53638b;
    border-radius: 13px;
    padding: 10px;
}

div[data-testid="stMetric"] {
    background: rgba(9, 15, 29, 0.55);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 15px;
}

hr {
    border-color: rgba(148, 163, 184, 0.15);
}

footer {
    visibility: hidden;
}

@media (max-width: 768px) {
    .hero {
        padding: 23px;
    }

    .hero h1 {
        letter-spacing: -1px;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ AI RESEARCH PROTOTYPE · COMPUTER VISION + NLP</div>
    <h1>Multimodal <span class="gradient-text">AI Studio</span></h1>
    <p>
        Explore visual facial-expression predictions and text sentiment
        analysis in one interactive workspace. Bring both modalities
        together to compare the model outputs.
    </p>
    <span class="badge">● SYSTEM READY · LOCAL PROTOTYPE</span>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_text_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

# =========================================================
# INPUTS
# =========================================================

st.markdown(
    '<div class="section-heading">01 / Input workspace</div>'
    '<div class="section-subtitle">Provide one or both inputs to begin your analysis.</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2, gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">📷 Visual Intelligence</div>
        <div class="panel-description">
            Upload a clear face image for facial-expression classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        key="face_image"
    )

    if uploaded_image is not None:
        preview = Image.open(uploaded_image).convert("RGB")
        st.image(
            preview,
            caption="Image ready for analysis",
            use_container_width=True
        )

with right:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">💬 Text Intelligence</div>
        <div class="panel-description">
            Enter a sentence or short paragraph for sentiment classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    user_text = st.text_area(
        "Text to analyse",
        placeholder="Example: I'm really happy with the progress I've made today!",
        height=170,
        label_visibility="collapsed",
        key="sentiment_text"
    )

    st.caption("English text · Maximum 512 characters recommended")

st.write("")

analyse = st.button("✦  Analyse Inputs", use_container_width=True)

# =========================================================
# ANALYSIS ENGINE
# =========================================================

if analyse:

    if uploaded_image is None and not user_text.strip():
        st.warning("Please upload a face image or enter some text.")
        st.stop()

    image_result = None
    text_result = None

    # ---------------- TEXT ANALYSIS ----------------

    if user_text.strip():
        with st.spinner("Running text sentiment analysis..."):
            try:
                model = load_text_model()
                prediction = model(user_text[:512])[0]

                text_result = {
                    "label": prediction["label"],
                    "score": float(prediction["score"])
                }

            except Exception as error:
                st.error(
                    "Text analysis failed. Check your internet connection "
                    "if the model is being downloaded for the first time."
                )
                st.exception(error)

    # ---------------- IMAGE ANALYSIS ----------------

    if uploaded_image is not None:
        temp_path = None

        with st.spinner("Analysing visible facial expressions..."):
            try:
                image = Image.open(uploaded_image).convert("RGB")

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False
                ) as temp_file:
                    image.save(temp_file, format="JPEG")
                    temp_path = temp_file.name

                result = DeepFace.analyze(
                    img_path=temp_path,
                    actions=["emotion"],
                    enforce_detection=True,
                    detector_backend="opencv"
                )

                if isinstance(result, list):
                    result = result[0]

                emotions = result.get("emotion", {})
                dominant = result.get("dominant_emotion", "Unknown")

                image_result = {
                    "dominant": dominant.capitalize(),
                    "emotions": emotions
                }

            except ValueError:
                st.warning(
                    "No face was detected. Try a clear, front-facing image."
                )

            except Exception as error:
                st.error("Facial-expression analysis failed.")
                st.exception(error)

            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    # Save results for display
    st.session_state["image_result"] = image_result
    st.session_state["text_result"] = text_result
    st.session_state["analysis_done"] = True

# =========================================================
# RESULTS DASHBOARD
# =========================================================

if st.session_state.get("analysis_done", False):

    image_result = st.session_state.get("image_result")
    text_result = st.session_state.get("text_result")

    st.divider()

    st.markdown("""
    <div class="section-heading">02 / Intelligence dashboard</div>
    <div class="section-subtitle">
        Predictions generated from the supplied image and text.
    </div>
    """, unsafe_allow_html=True)

    result_left, result_right = st.columns(2, gap="large")

    # ---------------- FACIAL RESULTS ----------------

    with result_left:
        st.markdown("""
        <div class="panel-title">📷 Visual analysis</div>
        """, unsafe_allow_html=True)

        if image_result:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Top expression prediction</div>
                    <div class="result-value">{image_result["dominant"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            sorted_emotions = sorted(
                image_result["emotions"].items(),
                key=lambda item: item[1],
                reverse=True
            )

            for emotion, score in sorted_emotions:
                st.write(
                    f"**{emotion.capitalize()}** · {float(score):.2f}%"
                )
                st.progress(
                    min(max(float(score) / 100, 0.0), 1.0)
                )

        else:
            st.info("No successful facial-expression prediction is available.")

    # ---------------- TEXT RESULTS ----------------

    with result_right:
        st.markdown("""
        <div class="panel-title">💬 Text analysis</div>
        """, unsafe_allow_html=True)

        if text_result:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted sentiment</div>
                    <div class="result-value">{text_result["label"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Model confidence",
                f'{text_result["score"]:.2%}'
            )

            st.progress(text_result["score"])

        else:
            st.info("No successful text sentiment prediction is available.")

    # ---------------- MULTIMODAL FUSION ----------------

    st.divider()

    st.markdown("""
    <div class="section-heading">03 / Multimodal comparison</div>
    <div class="section-subtitle">
        Compare the outputs from both models. This prototype uses a
        transparent rule-based comparison, not a trained fusion model.
    </div>
    """, unsafe_allow_html=True)

    if image_result and text_result:
        expression = image_result["dominant"].lower()
        sentiment = text_result["label"].upper()

        positive_expressions = {"happy"}
        negative_expressions = {"sad", "angry", "fear", "disgust"}

        if expression in positive_expressions and sentiment == "POSITIVE":
            comparison = "Outputs are directionally aligned"
            detail = (
                "The facial-expression model predicted Happy and the "
                "text model predicted Positive."
            )
        elif expression in negative_expressions and sentiment == "NEGATIVE":
            comparison = "Outputs are directionally aligned"
            detail = (
                f"The facial-expression model predicted {expression.title()} "
                "and the text model predicted Negative."
            )
        else:
            comparison = "Outputs differ or are inconclusive"
            detail = (
                f"Visual prediction: {expression.title()} · "
                f"Text prediction: {sentiment.title()}. "
                "Different outputs do not necessarily indicate a problem."
            )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Cross-modal comparison</div>
                <div class="result-value" style="font-size:23px;">
                    {comparison}
                </div>
                <p class="small-note">{detail}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif image_result:
        st.info(
            "Visual analysis is complete. Add text and analyse again "
            "to compare both modalities."
        )

    elif text_result:
        st.info(
            "Text analysis is complete. Upload a face image and analyse "
            "again to compare both modalities."
        )

    else:
        st.warning(
            "No successful predictions are available to compare. "
            "Please check the input and try again."
        )

    st.caption(
        "Research note: facial-expression and sentiment predictions are "
        "imperfect model outputs. They do not establish a person's actual "
        "emotional state. The comparison is a simple prototype rule."
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

footer_left, footer_right = st.columns([3, 1])

with footer_left:
    st.markdown(
        '<div class="small-note">'
        '🧠 <b>Multimodal AI Studio</b> · Academic research prototype'
        '<br>Computer Vision + Natural Language Processing'
        '</div>',
        unsafe_allow_html=True
    )

with footer_right:
    st.markdown(
        '<div class="small-note" style="text-align:right;">'
        'Built with Streamlit'
        '</div>',
        unsafe_allow_html=True
    )