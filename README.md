# SignPay — Behavioral Signature Authentication for UPI

A working prototype that replaces the 4-digit UPI PIN with a handwritten signature using on-device behavioral biometrics.

**Problem:** Every UPI scam exploits the static, visible, typeable PIN (only 10,000 combinations for 4 digit upi no.).

**Solution:** Draw your unique signature instead. The system captures:
- Shape (pixel pattern)
- Timing (stroke speed, pauses)
- Pressure (capacitive touch data)

Even if someone watches your screen, they can't replicate HOW you draw.

## Features

✅ Canvas-based signature capture (touch & mouse)  
✅ Python backend with DTW-inspired timing matching  
✅ Pixel similarity + behavioral biometric comparison  
✅ On-device .sgpx file storage (never transmitted)  
✅ Real-time matching visualization  
✅ Fallback to local matching if backend offline  

## Tech Stack

- **Frontend:** Vanilla JavaScript + Canvas API
- **Backend:** Python Flask + Pillow + NumPy
- **Matching:** Custom DTW algorithm + SSIM-inspired pixel comparison
- **Storage:** JSON .sgpx files (binary format planned) - (a new file type to be defined)


## Demo Video

Watch the demo:

https://github.com/manishkumarkmmani-star/signpay-upi-auth/raw/main/demo.mp4


## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run Backend
```bash
python app.py
```

### Open Frontend
Open `index.html` in Chrome

### Demo Flow
1. Set signature (draw once)
2. Confirm signature (draw again)
3. Send payment (enter amount)
4. Authenticate (draw to authorize)
5. Watch real-time matching score

## How Matching Works

**Pixel Similarity (65% weight)**
- Canvas PNG → downsample to 64×32 binary grid
- Compare ink pixels between enrollment & test
- Returns 0-100% shape similarity

**Timing Similarity (35% weight)**
- Extract x,y,timestamp vectors from strokes
- Normalize to 0-1 range
- Compare using sampling + Euclidean distance
- Returns 0-100% behavioral similarity

**Final Score = (Pixel × 0.65) + (Timing × 0.35)**  
**Threshold = 52% for authentication**

## API Endpoints

| Method | Route   | Purpose |
|--------|---------|---------|
| POST   | /enroll | Save signature as .sgpx |
| POST   | /verify | Match new drawing vs stored |
| POST   | /reset  | Delete old, save new signature |
| GET    | /status | Check if user enrolled |
| GET    | /sgpx   | View .sgpx metadata |

## Project Structure
signpay-upi-auth/
├── app.py                # Flask server
├── signature_engine.py   # Core matching logic
├── index.html            # Interactive frontend
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License


## Security Model

🔒 **On-Device Only** — .sgpx file never leaves your phone  
📂 **Proprietary Format** — Unpublished .sgpx spec adds security layer  
🔐 **AES-256 Ready** — Encryption key derived from device ID  
⚡ **Behavioral Auth** — Shape + timing + pressure = unbreakable  
🚫 **No PIN Entry** — Signature replaces typed PIN entirely  

## Next Steps

- [ ] Implement AES-256 encryption for .sgpx files
- [ ] Convert .sgpx from JSON to proprietary binary format
- [ ] Add device-ID binding (hardware serial)
- [ ] Build Android/iOS native apps with native signing APIs
- [ ] Integrate with actual UPI backend
- [ ] Deploy as NPCI-compatible additional auth layer

## Known Limitations

- Currently uses basic pixel downsampling (no advanced ML)
- Timing data accuracy depends on browser/device precision
- .sgpx files currently JSON (not encrypted)
- Prototype-level UI (not production-ready styling)

## Author

**Manish Kumar K M**  
ECE Final Year Student | Bangalore  
Original concept & implementation  (Build phase)

## License & Usage

This project is protected by copyright. 
- ✅ You can VIEW the source code
- ❌ You cannot USE it without permission
- ❌ You cannot MODIFY or DISTRIBUTE it

This is intentional IP protection. SignPay is a proprietary concept.

For licensing inquiries or collaboration, contact: [your email]

## Disclaimer

This is a working prototype for educational purposes. Not suitable for production UPI systems without significant security hardening and regulatory compliance.

---

**Built for:** Paytm Ideathon 2026  
**Created:** May 2026  
**Status:** Active Development
