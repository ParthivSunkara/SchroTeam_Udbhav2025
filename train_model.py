# made by 3 humans on mars lol
import os
import random
import time
from faker import Faker
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from scipy.sparse import hstack, csr_matrix
import joblib

N_PHISH = 250
N_BENIGN = 250
OUT_CSV = "data/emails_500.csv"
MODEL_OUT = "models/model_enhanced.joblib"
VEC_OUT = "models/vectorizer_enhanced.joblib"
RANDOM_SEED = 42

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

PHISH_TEMPLATES = [
    "Urgent: {action} your {service} account now: {link}",
    "Your {service} account will be suspended. {action} here: {link}",
    "We detected unusual login from {city}. {action} to secure: {link}",
    "{service} payment failed. {action} to verify payment: {link}",
    "Security alert: {service} requested password reset. {action}: {link}",
    "Claim your {reward} now. Limited time: {link}",
    "Your delivery is on hold. {action} to release: {link}",
    "Please confirm your identity for {service}: {link}",
    "Important: update billing info for {service}: {link}",
    "Temporary password: {pw}. Use it to {action}: {link}"
]

BENIGN_TEMPLATES = [
    "Meeting moved to {time}",
    "Lunch tomorrow? {place}",
    "Invoice for last month attached",
    "Reminder: {event} starts soon",
    "Please review the attached file about {topic}",
    "Thanks for your help earlier regarding {topic}",
    "Update on project status: {topic}",
    "Here’s the document you asked for",
    "Schedule for next week - confirm availability",
    "Happy birthday! 🎉 Hope you have a great day"
]

ACTIONS = ["verify", "confirm", "update", "authenticate", "reset"]
SERVICES = ["bank", "email", "payroll", "portal", "account"]
CITIES = ["Delhi", "London", "New York", "Mumbai", "Berlin"]
REWARDS = ["gift card", "cashback", "reward", "voucher"]
PLACES = ["cafeteria", "office", "meeting room"]
TOPICS = ["Q3 figures", "client feedback", "timeline", "budget"]

URL_RE = re.compile(r'https?://[^\s]+', re.IGNORECASE)
IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')

URGENCY_WORDS = {"urgent","immediately","now","verify","action required","suspend","suspended","locked","confirm","asap","important"}
MONEY_WORDS = {"free","prize","gift","winner","claim","redeem","reward","cash","card","voucher"}
SECURITY_WORDS = {"password","account","login","credentials","credential","reset","authenticate","verify"}

def extract_basic_features(text):
    text = "" if text is None else str(text)
    text_low = text.lower()
    # counts
    urls = URL_RE.findall(text)
    num_urls = len(urls)
    num_ip = len(IP_RE.findall(text))
    num_emails = len(EMAIL_RE.findall(text))
    urgency_count = sum(text_low.count(w) for w in URGENCY_WORDS)
    money_count = sum(text_low.count(w) for w in MONEY_WORDS)
    security_count = sum(text_low.count(w) for w in SECURITY_WORDS)

    letters = re.findall(r'[A-Za-z]', text)
    uppercase_letters = sum(1 for c in letters if c.isupper())
    uppercase_ratio = (uppercase_letters / len(letters)) if letters else 0.0

    exclam = text.count('!')
    question = text.count('?')

    length = len(text)
    token_count = len(re.findall(r'\w+', text))

    suspicious_chars = sum(1 for c in text if c in {'$', '%', '^', '*', '&', '~', '`', '|', '<', '>'})

    has_click = 1 if "click" in text_low or "verify" in text_low or "reset" in text_low else 0

    return np.array([
        num_urls, num_ip, num_emails,
        urgency_count, money_count, security_count,
        uppercase_ratio, exclam, question,
        length, token_count, suspicious_chars,
        has_click
    ], dtype=float)

def extract_features_for_list(texts):
    feats = [extract_basic_features(t) for t in texts]
    return np.vstack(feats)


def make_phish_instance(idx):
    tpl = random.choice(PHISH_TEMPLATES)
    link = f"https://{fake.domain_name()}/{fake.word()}{random.randint(1,999)}"
    text = tpl.format(
        action=random.choice(ACTIONS),
        service=random.choice(SERVICES),
        link=link,
        city=random.choice(CITIES),
        reward=random.choice(REWARDS),
        pw=f"{random.randint(1000,9999)}",
    )
    if random.random() < 0.4:
        text = f"Hi {fake.first_name()}, " + text
    if random.random() < 0.3:
        text = text + f" - {fake.company()} Team"
    return text

def make_benign_instance(idx):
    tpl = random.choice(BENIGN_TEMPLATES)
    text = tpl.format(
        time=f"{random.randint(9,17)}:{random.choice(['00','30'])}",
        place=random.choice(PLACES),
        event=random.choice(["training","meeting","lecture"]),
        topic=random.choice(TOPICS)
    )
    if random.random() < 0.5:
        text = text + " Please confirm when you can."
    if random.random() < 0.2:
        text = f"Hi {fake.first_name()}, " + text
    return text

def generate_dataset(n_phish=N_PHISH, n_benign=N_BENIGN, out_csv=OUT_CSV):
    rows = []
    idx = 0
    for _ in range(n_phish):
        rows.append({"id": idx, "text": make_phish_instance(idx), "label": 1})
        idx += 1
    for _ in range(n_benign):
        rows.append({"id": idx, "text": make_benign_instance(idx), "label": 0})
        idx += 1
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")
    return df

def train_from_csv(csv_path=OUT_CSV, test_size=0.2, random_state=RANDOM_SEED):
    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path)
    texts = df['text'].fillna('').astype(str).values
    y = df['label'].astype(int).values

    print("Extracting TF-IDF (unigrams + bigrams)...")
    vec = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    X_text = vec.fit_transform(texts)

    print("Extracting numeric features...")
    X_num = extract_features_for_list(texts)
    X_num_sp = csr_matrix(X_num)

    print("Combining features...")
    X = hstack([X_text, X_num_sp], format='csr')

    print("Train/test split...")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    print("Training LogisticRegression (this may take a little while depending on environment)...")
    start = time.time()
    clf = LogisticRegression(
        solver='saga',
        max_iter=2000,
        C=1.0,
        class_weight='balanced',
        n_jobs=-1,
        random_state=random_state
    )
    clf.fit(Xtr, ytr)
    elapsed = time.time() - start
    print(f"Trained in {elapsed:.2f}s")

    print("Predicting on test set...")
    preds = clf.predict(Xte)
    acc = accuracy_score(yte, preds)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(yte, preds, digits=4))

    p, r, f1, _ = precision_recall_fscore_support(yte, preds, average='binary', pos_label=1)
    print(f"Phish precision: {p:.4f}, recall: {r:.4f}, f1: {f1:.4f}")

    print("Saving model and vectorizer...")
    joblib.dump(clf, MODEL_OUT)
    joblib.dump(vec, VEC_OUT)
    print("Saved to:", MODEL_OUT, VEC_OUT)

    return clf, vec, Xtr.shape, Xte.shape, (acc, p, r, f1)

if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset()
    print("Sample rows:")
    print(df.sample(6).to_string(index=False))
    print("\nStarting training pipeline...")
    clf, vec, train_shape, test_shape, metrics = train_from_csv()
    print("Done. Train shape:", train_shape, "Test shape:", test_shape)
    print("Metrics (acc, prec, rec, f1):", metrics)
    print("You can now use detector.py to load models/model_enhanced.joblib and models/vectorizer_enhanced.joblib")