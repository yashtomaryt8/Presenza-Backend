Place the following two model files in this directory:

  2.7_80x80_MiniFASNetV2.pth
  4_0_0_80x80_MiniFASNetV1SE.pth

Download automatically:
  cd backend
  source venv/bin/activate
  python api/anti_spoof.py --download

Manual download URLs:
  https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth
  https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth

WITHOUT these files: system uses LBP texture fallback (still blocks most photo attacks).
WITH these files: deep neural net liveness detection (blocks photos, screens, masks).
