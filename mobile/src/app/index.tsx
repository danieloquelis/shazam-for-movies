import { CameraView, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';
import { router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import React from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import { ScanRing } from '@/components/scan-ring';
import { queryClip } from '@/lib/api';
import { SCAN_DURATION_SEC } from '@/lib/config';

type Phase = 'idle' | 'recording' | 'uploading';

export default function ScanScreen() {
  // ---- hooks (must always run, in the same order) -------------------------
  const [cameraPerm, requestCameraPerm] = useCameraPermissions();
  // expo-camera requires mic permission to record video, even though the
  // backend never reads the audio track.
  const [micPerm, requestMicPerm] = useMicrophonePermissions();

  const cameraRef = React.useRef<CameraView | null>(null);
  const stopTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const [phase, setPhase] = React.useState<Phase>('idle');
  const [errorMsg, setErrorMsg] = React.useState<string | null>(null);

  const startScan = React.useCallback(async () => {
    if (!cameraRef.current || phase !== 'idle') return;
    setErrorMsg(null);
    setPhase('recording');

    // Schedule the auto-stop. recordAsync resolves AFTER stopRecording
    // is called, with the final video URI.
    stopTimerRef.current = setTimeout(() => {
      cameraRef.current?.stopRecording();
    }, SCAN_DURATION_SEC * 1000);

    try {
      const recording = await cameraRef.current.recordAsync({
        // Backstop in case the timer above misfires.
        maxDuration: SCAN_DURATION_SEC + 2,
      });

      if (!recording?.uri) {
        setPhase('idle');
        setErrorMsg('Recording failed. Try again.');
        return;
      }

      setPhase('uploading');
      const result = await queryClip(recording.uri);

      switch (result.status) {
        case 'match':
          router.push({
            pathname: '/result',
            params: {
              title: result.title,
              timestampHuman: result.timestampHuman,
              timestampSec: String(result.timestampSec),
              confidence: String(result.confidence),
              visualScore: String(result.visualScore),
            },
          });
          break;
        case 'no_match':
          router.push({ pathname: '/result', params: { miss: '1' } });
          break;
        case 'error':
          setErrorMsg(result.message);
          break;
      }
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
      setPhase('idle');
    }
  }, [phase]);

  React.useEffect(() => {
    return () => {
      if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
    };
  }, []);

  // ---- single render decision (after every hook has been called) ---------

  if (!cameraPerm || !micPerm) {
    // Hooks haven't resolved yet on first render — keep it neutral.
    return <View style={styles.container} />;
  }

  if (!cameraPerm.granted || !micPerm.granted) {
    return (
      <View style={styles.permsContainer}>
        <StatusBar style="light" />
        <Text style={styles.permsTitle}>Camera access</Text>
        <Text style={styles.permsBody}>
          To identify a movie, the app needs access to the camera (to record a short clip) and the
          microphone (a system requirement for video capture — audio is not used).
        </Text>
        <TouchableOpacity
          style={styles.permsButton}
          onPress={async () => {
            await requestCameraPerm();
            await requestMicPerm();
          }}
        >
          <Text style={styles.permsButtonText}>Grant access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar style="light" />
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} mode="video" facing="back" />

      {/* Subtle vignette so the button is always legible */}
      <View pointerEvents="none" style={styles.vignette} />

      <View style={styles.headerArea}>
        <Text style={styles.appTitle}>Shazam for Movies</Text>
        <Text style={styles.hint}>
          {phase === 'idle' && 'Point at a screen and tap to identify the scene'}
          {phase === 'recording' && `Hold still — recording ${SCAN_DURATION_SEC}s`}
          {phase === 'uploading' && 'Matching against the index…'}
        </Text>
      </View>

      <View style={styles.buttonArea}>
        {errorMsg && <Text style={styles.error}>{errorMsg}</Text>}

        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Start scan"
          accessibilityState={{ disabled: phase !== 'idle' }}
          disabled={phase !== 'idle'}
          onPress={startScan}
          style={({ pressed }) => [styles.scanWrap, pressed && phase === 'idle' && { opacity: 0.85 }]}
        >
          <ScanRing
            active={phase === 'recording'}
            durationMs={SCAN_DURATION_SEC * 1000}
            size={220}
          />
          <View style={styles.innerButton}>
            {phase === 'uploading' ? (
              <ActivityIndicator color="#000" />
            ) : (
              <Text style={styles.innerLabel}>{phase === 'recording' ? 'REC' : 'SCAN'}</Text>
            )}
          </View>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  vignette: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.18)',
  },
  headerArea: {
    position: 'absolute',
    top: 64,
    left: 24,
    right: 24,
    alignItems: 'center',
    gap: 6,
  },
  appTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  hint: {
    color: 'rgba(255,255,255,0.75)',
    fontSize: 14,
    textAlign: 'center',
  },
  buttonArea: {
    position: 'absolute',
    bottom: 80,
    left: 0,
    right: 0,
    alignItems: 'center',
    gap: 16,
  },
  error: {
    color: '#ff6b6b',
    fontSize: 13,
    paddingHorizontal: 24,
    textAlign: 'center',
  },
  scanWrap: {
    width: 240,
    height: 240,
    alignItems: 'center',
    justifyContent: 'center',
  },
  innerButton: {
    position: 'absolute',
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  innerLabel: {
    color: '#000',
    fontSize: 22,
    fontWeight: '700',
    letterSpacing: 1,
  },
  permsContainer: {
    flex: 1,
    backgroundColor: '#000',
    padding: 32,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 16,
  },
  permsTitle: {
    color: '#fff',
    fontSize: 22,
    fontWeight: '600',
  },
  permsBody: {
    color: 'rgba(255,255,255,0.75)',
    fontSize: 15,
    textAlign: 'center',
    lineHeight: 22,
  },
  permsButton: {
    marginTop: 12,
    backgroundColor: '#fff',
    paddingHorizontal: 28,
    paddingVertical: 14,
    borderRadius: 999,
  },
  permsButtonText: {
    color: '#000',
    fontWeight: '600',
    fontSize: 16,
  },
});
