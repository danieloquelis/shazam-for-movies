import type { ExpoConfig, ConfigContext } from 'expo/config';

/**
 * Dynamic Expo config. Reads EXPO_PUBLIC_* env vars at build/start time so
 * different builds can target different backends. See `.env.example`.
 */
export default ({ config }: ConfigContext): ExpoConfig => ({
  ...config,
  name: 'whomie',
  slug: 'whomie',
  version: '1.0.0',
  orientation: 'portrait',
  icon: './assets/images/icon.png',
  scheme: 'whomie',
  userInterfaceStyle: 'automatic',
  ios: {
    icon: './assets/expo.icon',
    infoPlist: {
      NSCameraUsageDescription:
        'Whomie uses the camera to record a short clip of the screen so it can identify the movie.',
      NSMicrophoneUsageDescription:
        'A microphone permission is required for video capture, but audio is not used by the matcher.',
      NSPhotoLibraryUsageDescription:
        'Used only if you choose to import a clip from your library instead of recording one.',
    },
  },
  android: {
    adaptiveIcon: {
      backgroundColor: '#000000',
      foregroundImage: './assets/images/android-icon-foreground.png',
      backgroundImage: './assets/images/android-icon-background.png',
      monochromeImage: './assets/images/android-icon-monochrome.png',
    },
    predictiveBackGestureEnabled: false,
    permissions: ['android.permission.CAMERA', 'android.permission.RECORD_AUDIO'],
  },
  web: {
    output: 'static',
    favicon: './assets/images/favicon.png',
  },
  plugins: [
    'expo-router',
    [
      'expo-camera',
      {
        cameraPermission: 'Whomie uses the camera to record a short clip of the screen.',
        microphonePermission:
          'Required by the OS for video recording — audio is not used by the matcher.',
        recordAudioAndroid: true,
      },
    ],
    [
      'expo-splash-screen',
      {
        backgroundColor: '#000000',
        android: {
          image: './assets/images/splash-icon.png',
          imageWidth: 76,
        },
      },
    ],
  ],
  experiments: {
    typedRoutes: true,
    reactCompiler: true,
  },
  extra: {
    apiUrl: process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000',
    apiKey: process.env.EXPO_PUBLIC_API_KEY ?? 'dev-key-change-me',
  },
});
