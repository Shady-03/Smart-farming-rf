import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN slt_data.csv
# ─────────────────────────────────────────────
print("📂 Loading slt_data.csv...")
data = pd.read_csv("datasets/slt_data.csv")

data['label']     = data['label'].str.strip().str.lower()
data['season']    = data['season'].str.strip().str.capitalize()
data['state']     = data['state'].str.strip().str.title()
data['soil type'] = data['soil type'].str.strip().str.lower()
data = data.drop_duplicates()

print(f"   Shape        : {data.shape}")
print(f"   Crops        : {data['label'].nunique()} → {sorted(data['label'].unique())}")
print(f"   Seasons      : {data['season'].unique().tolist()}")
print(f"   States       : {data['state'].unique().tolist()}")
print(f"   Soil types   : {data['soil type'].unique().tolist()}")

# ─────────────────────────────────────────────
# 2. ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────
season_enc = LabelEncoder()
state_enc  = LabelEncoder()
soil_enc   = LabelEncoder()
label_enc  = LabelEncoder()

data['season_enc'] = season_enc.fit_transform(data['season'])
data['state_enc']  = state_enc.fit_transform(data['state'])
data['soil_enc']   = soil_enc.fit_transform(data['soil type'])

# ─────────────────────────────────────────────
# 3. FEATURES & LABELS
# ─────────────────────────────────────────────
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
            'season_enc', 'state_enc', 'soil_enc']

X = data[FEATURES].values
y = label_enc.fit_transform(data['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n   Train samples : {len(X_train)}")
print(f"   Test  samples : {len(X_test)}")

# ─────────────────────────────────────────────
# 4. DEFINE ALL MODELS
# ─────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),
    "SVM": SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=42
    ),
}

# ─────────────────────────────────────────────
# 5. TRAIN & BENCHMARK ALL MODELS
# ─────────────────────────────────────────────
print("\n🚀 Training and benchmarking all models...\n")
print(f"{'Model':<22} {'Test Acc':>10}  {'CV Mean':>10}  {'CV Std':>8}  {'Status'}")
print("─" * 70)

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_sc))
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5,
                                scoring='accuracy', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std  = cv_scores.std()

    status = "🏆 BEST" if len(results) == 0 else ""
    print(f"{name:<22} {test_acc*100:>9.2f}%  {cv_mean*100:>9.2f}%  {cv_std*100:>7.2f}%")

    results.append((name, test_acc, cv_mean, cv_std, model))
    trained_models[name] = model

# ─────────────────────────────────────────────
# 6. PICK BEST MODEL (by CV mean accuracy)
# ─────────────────────────────────────────────
results.sort(key=lambda x: x[2], reverse=True)
best_name, best_test, best_cv, best_std, best_model = results[0]

print(f"\n🏆 Best Model  : {best_name}")
print(f"   Test Acc    : {best_test*100:.2f}%")
print(f"   CV Acc      : {best_cv*100:.2f}% ± {best_std*100:.2f}%")

# ─────────────────────────────────────────────
# 7. FULL CLASSIFICATION REPORT
# ─────────────────────────────────────────────
y_pred = best_model.predict(X_test_sc)
print(f"\n📊 Classification Report — {best_name}:")
print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

# ─────────────────────────────────────────────
# 8. SANITY CHECK — Karnataka Kharif
# ─────────────────────────────────────────────
print("\n── Sanity Check: Karnataka + Kharif ──")
filtered = data[(data['state'] == 'Karnataka') & (data['season'] == 'Kharif')]
unique_crops = filtered['label'].unique()

sanity_results = []
for crop_name in unique_crops:
    crop_rows = filtered[filtered['label'] == crop_name]
    avg = crop_rows[['N','P','K','temperature','humidity','ph','rainfall']].mean()

    s_enc = season_enc.transform(['Kharif'])[0]
    st_enc = state_enc.transform(['Karnataka'])[0]
    # use most common soil for this crop+state
    common_soil = crop_rows['soil type'].mode()[0]
    sl_enc = soil_enc.transform([common_soil])[0]

    input_vec = np.array([[
        avg['N'], avg['P'], avg['K'],
        avg['temperature'], avg['humidity'], avg['ph'], avg['rainfall'],
        s_enc, st_enc, sl_enc
    ]])
    input_sc = scaler.transform(input_vec)
    probs = best_model.predict_proba(input_sc)[0]

    if crop_name in label_enc.classes_:
        idx = label_enc.transform([crop_name])[0]
        score = probs[idx]
        sanity_results.append((crop_name, round(score * 100, 2)))

sanity_results.sort(key=lambda x: x[1], reverse=True)
print(f"{'Crop':<15} {'Score':>8}   Bar")
print("─" * 45)
for crop, pct in sanity_results:
    bar = "█" * int(pct / 3)
    print(f"  {crop:<15} {pct:6.2f}%   {bar}")

# ─────────────────────────────────────────────
# 9. SAVE EVERYTHING
# ─────────────────────────────────────────────
joblib.dump(best_model,  "crop_model.pkl")
joblib.dump(label_enc,   "label_encoder_model.pkl")
joblib.dump(scaler,      "feature_scaler.pkl")
joblib.dump(season_enc,  "season_encoder.pkl")
joblib.dump(state_enc,   "state_encoder.pkl")
joblib.dump(soil_enc,    "soil_encoder.pkl")

print(f"\n💾 Saved: crop_model.pkl  ({best_name})")
print("💾 Saved: label_encoder_model.pkl")
print("💾 Saved: feature_scaler.pkl")
print("💾 Saved: season_encoder.pkl")
print("💾 Saved: state_encoder.pkl")
print("💾 Saved: soil_encoder.pkl")

# ─────────────────────────────────────────────
# 10. HOW TO UPDATE app.py
# ─────────────────────────────────────────────
