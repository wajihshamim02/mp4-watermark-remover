# 🎬 Watermark Remover — Flutter + Python

## Features
✅ Auto watermark detection (AI)
✅ Manual region selection — khud box draw karo
✅ Audio preserved — awaz nahi jayegi
✅ Flutter desktop app (macOS/Windows/Linux)

---

## STEP 1 — ffmpeg install karo (ZAROOR hai audio ke liye)

```bash
brew install ffmpeg
```

Check karo:
```bash
ffmpeg -version
```

---

## STEP 2 — Backend chalao

```bash
cd wm_app/backend
pip3 install -r requirements.txt
python3 -m uvicorn main:app --reload --port 8000
```

Terminal band mat karo.

---

## STEP 3 — Flutter app chalao

```bash
cd wm_app/flutter_app

# Dependencies install karo
flutter pub get

# macOS desktop
flutter run -d macos
```

---

## App Use Karna

### Auto Mode (AI):
1. Video select karo
2. "Auto Detect" ON rakho
3. "Watermark Remove Karo" dabao
4. Done! Awaz bhi rahegi

### Manual Mode (khud select karo):
1. Video select karo
2. "Select" button dabao
3. Video frame pe box draw karo (jahan watermark ho)
4. Multiple boxes draw kar sakte ho
5. "Save Karo" dabao
6. "Watermark Remove Karo" dabao

---

## Audio Kaise Safe Hoti Hai

```
Original Video
     ↓
ffmpeg → Audio extract (aac)
     ↓
OpenCV → Video frames process (watermark remove)
     ↓
ffmpeg → Audio + Video merge back
     ↓
Final MP4 (with audio) ✅
```

---

## macOS Permission (agar file picker kaam na kare)

macOS/macos/Runner/Release.entitlements mein add karo:
```xml
<key>com.apple.security.files.user-selected.read-only</key>
<true/>
```
