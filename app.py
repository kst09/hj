from pathlib import Path
import json
import sys
import traceback
import zipfile

import streamlit as st

st.set_page_config(page_title="자동차 소리 감지", page_icon="🚗", layout="centered")
BASE = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def assets():
    print("[vehicle] importing inference dependencies", flush=True)
    from inference import load_assets
    print("[vehicle] loading model", flush=True)
    loaded = load_assets(BASE)
    print("[vehicle] model loaded", flush=True)
    return loaded


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def analyze(audio_bytes: bytes):
    model, config = assets()
    from inference import predict_audio
    print("[vehicle] preprocessing and predicting", flush=True)
    result = predict_audio(model, audio_bytes, config)
    print("[vehicle] prediction complete", flush=True)
    return result


print("[vehicle] app.py started; build=20261018-02; python=" + sys.version.split()[0], flush=True)
st.title("🚗 자동차 소리 감지")
st.caption("화면 버전: 20261018-02")
st.write("마이크로 약 10초간 녹음하거나 오디오 파일을 올려 차량 소리를 확인하세요.")
st.caption("녹음을 끝낸 뒤 분석합니다. 상시 실시간 감지 기능은 아닙니다.")

try:
    missing = [name for name in ("inference.py", "vehicle_config.json", "vehicle_model.keras")
               if not (BASE / name).is_file()]
    if missing:
        raise FileNotFoundError("GitHub에서 app.py와 같은 폴더에 올려주세요: " + ", ".join(missing))
    config = json.loads((BASE / "vehicle_config.json").read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or "threshold" not in config:
        raise ValueError("vehicle_config.json 내용이 모델 설정 형식이 아닙니다. 원본 파일로 교체하세요.")
    threshold_default = float(config["threshold"])
    if not 0.05 <= threshold_default <= 0.95:
        raise ValueError("설정 파일의 threshold를 확인하세요. 원본값은 0.5입니다.")
    if not zipfile.is_zipfile(BASE / "vehicle_model.keras"):
        raise ValueError("vehicle_model.keras가 정상 모델 파일이 아닙니다. 첨부했던 원본 모델을 다시 올려주세요.")
except Exception as exc:
    traceback.print_exc()
    st.error("앱 파일을 확인해 주세요.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("감지 설정")
    threshold = st.slider("차량 판정 기준", 0.05, 0.95, threshold_default, 0.05)
    st.caption("낮추면 더 자주 감지하고, 높이면 더 높은 점수일 때 감지합니다.")
    st.write("학습 입력: 16kHz · 10초 · 64×501 멜 스펙트로그램")

source = st.radio("입력 방법", ["마이크 녹음", "파일 업로드"], horizontal=True)
if source == "마이크 녹음":
    audio = st.audio_input("녹음 버튼을 누르고 약 10초 뒤 정지해 주세요.", sample_rate=16000)
else:
    audio = st.file_uploader("WAV / FLAC / OGG 파일", type=["wav", "flac", "ogg"])

if audio is None:
    st.info("오디오를 입력하면 결과가 여기에 표시됩니다.")
    st.stop()

if audio.size > 25 * 1024 * 1024:
    st.error("25MB 이하의 파일을 사용해 주세요.")
    st.stop()

st.audio(audio)
try:
    with st.spinner("모델을 불러와 분석하고 있습니다. 첫 분석은 잠시 걸릴 수 있습니다…"):
        result = analyze(audio.getvalue())
except Exception as exc:
    traceback.print_exc()
    st.error("분석 중 오류가 발생했습니다. 아래 오류 내용을 보내주세요.")
    st.exception(exc)
    st.stop()

if result["silent"]:
    st.warning("입력이 거의 무음입니다. 마이크 권한과 입력 장치를 확인해 주세요.")
if result["padded"]:
    st.warning(f"녹음이 {result['original_seconds']:.1f}초라서 뒤를 무음으로 채웠습니다. 10초 녹음을 권장합니다.")
if result["cropped"]:
    start = result["start_seconds"]
    st.info(f"전체 {result['original_seconds']:.1f}초 중 중앙 {start:.1f}–{start + 10:.1f}초만 분석했습니다.")

score = result["score"]
st.metric("차량 소리 점수", f"{score:.1%}")
st.progress(score)
if score >= threshold:
    st.success("차량 소리가 감지되었습니다.")
else:
    st.info("현재 기준에서는 차량 소리가 감지되지 않았습니다.")
st.caption(f"판정 기준: {threshold:.0%} · 이 점수는 검증된 정확도나 실제 거리 확률이 아닙니다.")

with st.expander("실제 모델에 넣은 오디오 확인"):
    st.audio(result["waveform"], sample_rate=result["sample_rate"])
    display_mel = result["mel_db"]
    display_mel = (display_mel - display_mel.min()) / max(float(display_mel.max() - display_mel.min()), 1e-6)
    st.image(display_mel[::-1, :], clamp=True,
             caption="멜 스펙트로그램 — 위쪽이 높은 주파수", width="stretch")

st.caption("이 앱은 차량 소리 유무를 분류합니다. 차량의 거리·접근 방향이나 충돌 위험은 판단하지 않습니다.")
