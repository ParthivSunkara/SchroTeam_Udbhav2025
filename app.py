# made by 3 humans on earth lol
import streamlit as st
import joblib

from sim_generator import generate_one, generate_bulk

# inside Streamlit UI (Generate tab)
st.header("Generate phishing templates / simulated emails")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    count = st.number_input("Count", min_value=1, max_value=500, value=10, step=1)
with col2:
    phishy_frac = st.slider("Phishy fraction", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
with col3:
    sophistication = st.slider("Phish sophistication (0 low - 1 high)", 0.0, 1.0, 0.6, 0.1)

if st.button("Generate samples"):
    df = generate_bulk(n=int(count), phishy_frac=float(phishy_frac), sophistication=float(sophistication), as_html=False)
    st.success(f"Generated {len(df)} samples")
    st.dataframe(df[["id","subject","label","severity","tags"]])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name="simulated_emails.csv", mime="text/csv")



# Load trained model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title + intro
st.markdown("<h1 style='text-align:center; color:#00BFFF;'>Phishing Email Detector 🔍</h1>", unsafe_allow_html=True)
st.write("Paste your email content below and let’s see if it’s safe:")

# Text input area
email = st.text_area("✉️ Email content")

# Analyze button
if st.button("Analyze"):
    if email.strip() == "":
        st.warning("Please enter an email to analyze.")
    else:
        # Vectorize the email text
        email_vec = vectorizer.transform([email])

        # Get probability that it's phishing
        prob = model.predict_proba(email_vec)[0][1]
        threshold = 0.7  # change between 0.6–0.8 if needed

        # Display result with confidence
        if prob > threshold:
            st.error(f"🚨 This looks like a phishing email! (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ This email seems safe. (Confidence: {1 - prob:.2f})")

# Footer
st.markdown("<hr><p style='text-align:center;'>Made by SchröTeam <hr><br><br></p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Team members:</p>", unsafe_allow_html=True)
st.markdown("<br><p style='text-align:left;'>Sunkara Parthiv</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:left;'>R.No.: S20250020362</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:left;'>Email: parthiv.s25@iiits.in</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:left;'>Role: Team leader, developing the model and generator</p>", unsafe_allow_html=True)
st.markdown("<br><p style='text-align:left;'>Mohit Vaibhav Ram Prattipati</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:left;'>R.No.: S20250010275</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:left;'>Email: mohitvaibhvram.p25@iiits.in</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:left;'>Role: Developing HTML and CSS elements</p>", unsafe_allow_html=True)
st.markdown("<br><p style='text-align:left;'>Andhavarapu Gayatri Devi</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:left;'>R.No.: S20250010015</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:left;'>Email: gayatridevi.a25@iiits.in</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:left;'>Role: Developing the main website via app.py</p>", unsafe_allow_html=True)