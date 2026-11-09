from pathlib import Path
import streamlit as st
from PIL import Image
import torch

from src.search_engine import ParentSearchEngine
from src.trait_analyzer import TraitAnalyzer

# Configure page layout and aesthetics
st.set_page_config(
    page_title="Deep Kinship AI - Tri-Subject Verification & Trait Attribution",
    page_icon="🧬",
    layout="wide",
)

# Custom card styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e1e24;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #00c9a7;
        margin-bottom: 1rem;
    }
    .trait-card {
        background-color: #18191f;
        border: 1px solid #2d2e38;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(
    show_spinner="Initializing Neural Engine & Precomputing Parent Embeddings..."
)
def get_engine():
    return ParentSearchEngine()


# Initialize engine and trait analyzer with MTCNN landmark detection
engine = get_engine()
analyzer = TraitAnalyzer(engine.model, engine.mtcnn, engine.device)

st.title("🧬 Tri-Subject Kinship Engine")
st.caption(
    "Siamese InceptionResnetV1 with Bilinear Genetic Fusion & Landmark-Guided Phenotype Attribution"
)

tabs = st.tabs(["⚖️ 1-to-1 Triplet Verification & Traits", "🔍 1-to-N Parent Search"])

# ----------------- TAB 1: 1-to-1 TRIPLET VERIFICATION & TRAITS -----------------
with tabs[0]:
    st.subheader("Direct Triplet Verification & Phenotype Breakdown")
    st.markdown(
        "Verify whether candidate parents and child form a biological triad, then attribute facial traits."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        f_up = st.file_uploader("Father Photo", type=["jpg", "jpeg", "png"], key="v_f")
    with col2:
        m_up = st.file_uploader("Mother Photo", type=["jpg", "jpeg", "png"], key="v_m")
    with col3:
        c_up = st.file_uploader("Child Photo", type=["jpg", "jpeg", "png"], key="v_c")

    if f_up and m_up and c_up:
        img_f = Image.open(f_up)
        img_m = Image.open(m_up)
        img_c = Image.open(c_up)

        # Show input previews
        st.write("")
        c_prev1, c_prev2, c_prev3 = st.columns(3)
        with c_prev1:
            st.image(img_f, caption="Father Input", use_container_width=True)
        with c_prev2:
            st.image(img_m, caption="Mother Input", use_container_width=True)
        with c_prev3:
            st.image(img_c, caption="Child Input", use_container_width=True)

        if st.button(
            "Verify Kinship & Analyze Traits", type="primary", use_container_width=True
        ):
            with st.spinner(
                "Extracting embeddings and detecting anatomical landmarks..."
            ):
                t_f = engine.extract_child_tensor(img_f)
                t_m = engine.extract_child_tensor(img_m)
                t_c = engine.extract_child_tensor(img_c)

                with torch.no_grad():
                    logit = engine.model(t_f, t_m, t_c)
                    prob = torch.sigmoid(logit).item()

                trait_results = analyzer.analyze_all(img_f, img_m, img_c)

            st.write("")
            if prob >= 0.5:
                st.success(f"### ✅ Kinship Confirmed (Confidence: {prob * 100:.2f}%)")
            else:
                st.error(
                    f"### ❌ Kinship Rejected (Confidence: {(1.0 - prob) * 100:.2f}%)"
                )
            st.progress(prob)

            st.divider()
            st.subheader("🧬 Genetic Trait Inheritance Breakdown")
            st.caption(
                "Visual anatomical comparison comparing isolated facial landmarks against both parents."
            )

            trait_icons = {
                "Eyes": "👁️",
                "Nose": "👃",
                "Mouth & Lips": "👄",
                "Hair & Forehead": "💇",
                "Skin Complexion": "✨",
            }

            for trait_name, data in trait_results.items():
                icon = trait_icons.get(trait_name, "🧬")
                dominant_parent = data["dominant"]
                dominant_score = data["dominant_pct"]

                with st.container():
                    st.markdown(
                        f"""
                        <div class="trait-card">
                            <h4 style="margin:0 0 6px 0;">{icon} {trait_name} — Primarily inherited from <b>{dominant_parent}</b> ({dominant_score}%)</h4>
                            <p style="margin:0 0 8px 0; color:#b0b0b0; font-size:14px;">
                                Father Contribution: <b>{data["father_pct"]}%</b>  |  Mother Contribution: <b>{data["mother_pct"]}%</b>
                            </p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    c_f, c_c, c_m = st.columns(3)
                    with c_f:
                        st.image(
                            data["crop_f"],
                            caption=f"Father's {trait_name}",
                            use_container_width=True,
                        )
                    with c_c:
                        st.image(
                            data["crop_c"],
                            caption=f"Child's {trait_name}",
                            use_container_width=True,
                        )
                    with c_m:
                        st.image(
                            data["crop_m"],
                            caption=f"Mother's {trait_name}",
                            use_container_width=True,
                        )

                    # Normalized bar: 0.0 = 100% Father, 1.0 = 100% Mother
                    st.progress(data["mother_pct"] / 100.0)
                    st.caption("◀ Closer to Father | Closer to Mother ▶")
                    st.write("")

# ----------------- TAB 2: 1-to-N PARENT SEARCH -----------------
with tabs[1]:
    st.subheader("Biological Parent Pair Retrieval")
    st.markdown(
        "Upload any photo of a child to search and rank the most probable biological parents in the gallery."
    )

    col_q, col_res = st.columns([1, 2], gap="large")

    with col_q:
        st.markdown("### Query Child")
        uploaded_child = st.file_uploader(
            "Upload Child Image", type=["jpg", "jpeg", "png"], key="search_child"
        )
        top_k = st.slider(
            "Number of Candidates (Top-K)", min_value=1, max_value=10, value=5
        )
        search_clicked = st.button(
            "Search Biological Parents", type="primary", use_container_width=True
        )

        if uploaded_child:
            query_img = Image.open(uploaded_child)
            st.image(query_img, caption="Query Child", use_container_width=True)

    with col_res:
        st.markdown("### Ranked Parent Candidates")
        if search_clicked and uploaded_child:
            with st.spinner("Executing vectorized bilinear search across gallery..."):
                results = engine.search(query_img, top_k=top_k)

            for item in results:
                score = item["score"]
                pct = score * 100
                status_color = "#00c9a7" if pct >= 50 else "#ff5252"

                with st.container():
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left-color: {status_color}">
                            <h4>Rank #{item["rank"]} — Match Confidence: {pct:.1f}% (Family: {item["family_id"]})</h4>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    c_f, c_m = st.columns(2)
                    with c_f:
                        if Path(item["father_path"]).exists():
                            st.image(
                                Image.open(item["father_path"]),
                                caption="Candidate Father",
                                use_container_width=True,
                            )
                    with c_m:
                        if Path(item["mother_path"]).exists():
                            st.image(
                                Image.open(item["mother_path"]),
                                caption="Candidate Mother",
                                use_container_width=True,
                            )
                    st.progress(min(max(score, 0.0), 1.0))
                    st.divider()
