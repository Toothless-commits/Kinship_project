from pathlib import Path
import streamlit as st
from PIL import Image

from src.search_engine import ParentSearchEngine
from src.trait_analyzer import TraitAnalyzer

st.set_page_config(
    page_title="Deep Kinship AI - Tri-Subject Verification & Trait Attribution",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e1e24;
        padding: 1.1rem 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #00c9a7;
        margin-bottom: 0.9rem;
    }
    .trait-card {
        background-color: #18191f;
        border: 1px solid #2d2e38;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .source-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background-color: #2b2d3a;
        color: #00c9a7;
        margin-left: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(
    show_spinner="Initializing Neural Engine & Precomputing TTA Parent Embeddings..."
)
def get_engine():
    return ParentSearchEngine()


engine = get_engine()
analyzer = TraitAnalyzer(engine.model, engine.mtcnn, engine.device)

st.title("🧬 Tri-Subject Kinship Engine")
st.caption(
    "Siamese InceptionResnetV1 with Symmetric Genetic Fusion, TTA & Landmark-Guided Phenotype Attribution"
)

tabs = st.tabs(
    [
        "⚖️ 1-to-1 Triplet Verification & Traits",
        "🔍 1-to-N Parent Search",
        "🖼️ Parent Gallery Manager",
    ]
)

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
        img_f = Image.open(f_up).convert("RGB")
        img_m = Image.open(m_up).convert("RGB")
        img_c = Image.open(c_up).convert("RGB")

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
                "Extracting TTA embeddings and detecting anatomical landmarks..."
            ):
                prob = engine.verify_triplet(img_f, img_m, img_c)
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

                    st.progress(data["mother_pct"] / 100.0)
                    st.caption("◀ Closer to Father | Closer to Mother ▶")
                    st.write("")

# ----------------- TAB 2: 1-to-N PARENT SEARCH -----------------
with tabs[1]:
    st.subheader("Biological Parent Pair Retrieval")

    custom_count = (
        int((engine.gallery_df["source"] == "Custom").sum())
        if not engine.gallery_df.empty
        else 0
    )
    fiw_count = (
        int((engine.gallery_df["source"] == "FIW Dataset").sum())
        if not engine.gallery_df.empty
        else 0
    )

    st.caption(
        f"Active Gallery Index: **{custom_count} Custom Parent Couples** | **{fiw_count} FIW Dataset Couples**"
    )

    col_q, col_res = st.columns([1, 2], gap="large")

    with col_q:
        st.markdown("### Query Child")
        uploaded_child = st.file_uploader(
            "Upload Child Image", type=["jpg", "jpeg", "png"], key="search_child"
        )
        scope_choice = st.selectbox(
            "Gallery Scope",
            options=["Custom Gallery Only", "All Galleries", "FIW Dataset Only"],
            index=0 if custom_count > 0 else 1,
        )
        top_k = st.slider(
            "Number of Candidates (Top-K)", min_value=1, max_value=10, value=5
        )
        search_clicked = st.button(
            "Search Biological Parents", type="primary", use_container_width=True
        )

        if uploaded_child:
            query_img = Image.open(uploaded_child).convert("RGB")
            st.image(query_img, caption="Query Child", use_container_width=True)

    with col_res:
        st.markdown("### Ranked Parent Candidates")
        if search_clicked:
            if not uploaded_child:
                st.warning("Upload a child image first.")
            else:
                scope_map = {
                    "All Galleries": "All",
                    "Custom Gallery Only": "Custom",
                    "FIW Dataset Only": "FIW Dataset",
                }
                with st.spinner("Executing vectorized TTA search across gallery..."):
                    results = engine.search(
                        query_img,
                        top_k=top_k,
                        source_filter=scope_map[scope_choice],
                    )

                if not results:
                    st.info(
                        "No parent couples found in the selected gallery scope. "
                        "Add couples in the **🖼️ Parent Gallery Manager** tab or switch scope."
                    )

                for item in results:
                    score = item["score"]
                    pct = score * 100
                    status_color = "#00c9a7" if pct >= 50 else "#ff5252"

                    f_label = (
                        f"Father: {item['father_name']}"
                        if item["father_name"]
                        else "Candidate Father"
                    )
                    m_label = (
                        f"Mother: {item['mother_name']}"
                        if item["mother_name"]
                        else "Candidate Mother"
                    )

                    with st.container():
                        st.markdown(
                            f"""
                            <div class="metric-card" style="border-left-color: {status_color}">
                                <h4 style="margin:0;">
                                    Rank #{item["rank"]} — Match Confidence: {pct:.1f}% (Family: {item["family_id"]})
                                    <span class="source-badge">{item["source"]}</span>
                                </h4>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        c_f, c_m = st.columns(2)
                        with c_f:
                            f_path = Path(item["father_path"])
                            if f_path.exists():
                                st.image(
                                    f_path.read_bytes(),
                                    caption=f_label,
                                    use_container_width=True,
                                )
                        with c_m:
                            m_path = Path(item["mother_path"])
                            if m_path.exists():
                                st.image(
                                    m_path.read_bytes(),
                                    caption=m_label,
                                    use_container_width=True,
                                )
                        st.progress(min(max(score, 0.0), 1.0))
                        st.divider()

# ----------------- TAB 3: PARENT GALLERY MANAGER -----------------
with tabs[2]:
    st.subheader("Manual Parent Gallery Builder")
    st.markdown(
        "Upload **1 to 3 front-facing photos** each for Father and Mother. Uploading 2–3 photos "
        "per parent builds an averaged identity prototype that noticeably increases kinship confidence."
    )

    with st.form("enroll_parent_form", clear_on_submit=True):
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            fam_id_input = st.text_input(
                "Family ID",
                value=engine.next_custom_family_id(),
                help="Unique identifier for this parent couple (e.g., FAM_001).",
            )
        with meta_col2:
            father_name_input = st.text_input(
                "Father's Name (Optional)", placeholder="e.g., Rajesh"
            )
        with meta_col3:
            mother_name_input = st.text_input(
                "Mother's Name (Optional)", placeholder="e.g., Sunita"
            )

        up_col1, up_col2 = st.columns(2)
        with up_col1:
            gal_f_ups = st.file_uploader(
                "Father Photo(s)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="gal_f",
            )
        with up_col2:
            gal_m_ups = st.file_uploader(
                "Mother Photo(s)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="gal_m",
            )

        submitted = st.form_submit_button(
            "➕ Align & Add Couple to Gallery",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not gal_f_ups or not gal_m_ups:
            st.error("Please upload at least one Father photo and one Mother photo.")
        else:
            with st.spinner("Aligning faces with MTCNN and building TTA prototypes..."):
                f_imgs = [Image.open(u).convert("RGB") for u in gal_f_ups]
                m_imgs = [Image.open(u).convert("RGB") for u in gal_m_ups]
                info = engine.add_parent_couple(
                    family_id=fam_id_input,
                    father_imgs=f_imgs,
                    mother_imgs=m_imgs,
                    father_name=father_name_input,
                    mother_name=mother_name_input,
                )

            st.success(
                f"Enrolled **{info['family_id']}** ({len(f_imgs)} father photo(s), "
                f"{len(m_imgs)} mother photo(s)) into the live search gallery."
            )
            if not info["father_detected"] or not info["mother_detected"]:
                st.warning(
                    "Low MTCNN face confidence on at least one photo; a centered square crop was used."
                )

            al_col1, al_col2 = st.columns(2)
            with al_col1:
                st.image(
                    info["father_aligned"][0],
                    caption=f"Aligned Father ({info['family_id']})",
                    width=160,
                )
            with al_col2:
                st.image(
                    info["mother_aligned"][0],
                    caption=f"Aligned Mother ({info['family_id']})",
                    width=160,
                )

    st.divider()
    st.subheader("Enrolled Custom Parent Couples")
    custom_gallery_df = engine.get_custom_gallery()

    if custom_gallery_df.empty:
        st.info(
            "No custom parent couples enrolled yet. Use the form above to add your first couple."
        )
    else:
        for _, row in custom_gallery_df.iterrows():
            fid = row["family_id"]
            f_title = row["father_name"] if row["father_name"] else "Father"
            m_title = row["mother_name"] if row["mother_name"] else "Mother"
            f_paths = [p for p in str(row["father_path"]).split("|") if p]
            m_paths = [p for p in str(row["mother_path"]).split("|") if p]

            with st.container():
                c_info, c_img1, c_img2, c_del = st.columns([2, 1, 1, 1])
                with c_info:
                    st.markdown(f"#### `{fid}`")
                    st.write(f"**Father:** {f_title} ({len(f_paths)} photo(s))")
                    st.write(f"**Mother:** {m_title} ({len(m_paths)} photo(s))")
                with c_img1:
                    if f_paths and Path(f_paths[0]).exists():
                        st.image(
                            Path(f_paths[0]).read_bytes(), caption=f_title, width=120
                        )
                with c_img2:
                    if m_paths and Path(m_paths[0]).exists():
                        st.image(
                            Path(m_paths[0]).read_bytes(), caption=m_title, width=120
                        )
                with c_del:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Remove", key=f"del_{fid}"):
                        engine.remove_parent_couple(fid)
                        st.rerun()
                st.divider()
