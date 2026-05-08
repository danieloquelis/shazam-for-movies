# Mobile app (`whomie`)

Expo / React Native client for the Shazam-for-Movies engine. Records a short
clip with the camera, uploads it to the backend's `/query` endpoint, and shows
the matched movie + timestamp.

## Status

Phase 1 (record-and-upload) — see [`../docs/CLIENT.md`](../docs/CLIENT.md) for
the broader plan.

## Running

```bash
yarn install                 # one-time
cp .env.example .env         # set EXPO_PUBLIC_API_URL to your backend
yarn start                   # opens Expo dev menu
```

The app is a **dev build**, not Expo Go — `expo-camera` recording works in
either, but later phases (screen capture, on-device CLIP) need native modules
that Go can't load.

## Backend URL

Set `EXPO_PUBLIC_API_URL` in `.env`:

| Target | Value |
|---|---|
| iOS Simulator | `http://localhost:8000` |
| iOS device | `http://<host-LAN-IP>:8000` (e.g. `http://192.168.1.42:8000`) |
| Android emulator | `http://10.0.2.2:8000` |
| Android device | `http://<host-LAN-IP>:8000` |

The backend must be reachable from the device — same Wi-Fi network for LAN.
`EXPO_PUBLIC_API_KEY` must match the backend's `API_KEY` env var.

## Layout

```
mobile/
├── app.config.ts              # dynamic Expo config — reads .env
├── .env.example               # copy → .env, edit values
└── src/
    ├── app/
    │   ├── _layout.tsx        # root Stack (scan + result)
    │   ├── index.tsx          # camera scan screen (entry)
    │   └── result.tsx         # match-result modal
    ├── components/
    │   └── scan-ring.tsx      # animated countdown ring around the button
    └── lib/
        ├── api.ts             # POST /query client
        └── config.ts          # runtime config from app.config.ts extra
```

## Scan flow

1. User taps the big circular button on the home screen.
2. `expo-camera` records 5 seconds (visible countdown ring).
3. The recording is uploaded to `POST /query` as multipart form data.
4. Result screen pushes as a bottom-sheet modal: title + timestamp + confidence
   pill, or a "no match" message.
