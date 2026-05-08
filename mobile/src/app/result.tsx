import { router, useLocalSearchParams } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

type Params = {
  miss?: string;
  title?: string;
  timestampHuman?: string;
  timestampSec?: string;
  confidence?: string;
  visualScore?: string;
};

function confidenceLabel(c: number): { text: string; color: string } {
  if (c >= 0.8) return { text: 'High confidence', color: '#27c46a' };
  if (c >= 0.55) return { text: 'Medium confidence', color: '#f0c93b' };
  return { text: 'Low confidence', color: '#ff6b6b' };
}

export default function ResultScreen() {
  const params = useLocalSearchParams<Params>();
  const isMiss = params.miss === '1';

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />

      {isMiss ? (
        <View style={styles.body}>
          <Text style={styles.missEmoji}>🤷</Text>
          <Text style={styles.missTitle}>No confident match</Text>
          <Text style={styles.missBody}>
            Either the film isn&apos;t in the index yet, or the capture conditions made it hard to
            recognize. Try again with the screen filling more of the frame.
          </Text>
        </View>
      ) : (
        <ResultBody params={params} />
      )}

      <TouchableOpacity style={styles.cta} onPress={() => router.back()}>
        <Text style={styles.ctaText}>Scan again</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

function ResultBody({ params }: { params: Params }) {
  const confidence = Number(params.confidence ?? '0');
  const conf = confidenceLabel(confidence);

  return (
    <View style={styles.body}>
      <Text style={styles.kicker}>MATCH FOUND</Text>
      <Text style={styles.title}>{params.title}</Text>
      <Text style={styles.timestamp}>at {params.timestampHuman}</Text>

      <View style={styles.metaRow}>
        <View style={styles.metaPill}>
          <View style={[styles.dot, { backgroundColor: conf.color }]} />
          <Text style={styles.metaText}>
            {conf.text} · {Math.round(confidence * 100)}%
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0b',
    paddingHorizontal: 28,
    justifyContent: 'space-between',
  },
  body: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 12,
  },
  kicker: {
    color: 'rgba(255,255,255,0.55)',
    fontSize: 12,
    letterSpacing: 1.5,
    fontWeight: '600',
  },
  title: {
    color: '#fff',
    fontSize: 28,
    fontWeight: '700',
    textAlign: 'center',
    paddingHorizontal: 8,
  },
  timestamp: {
    color: 'rgba(255,255,255,0.65)',
    fontSize: 16,
    fontVariant: ['tabular-nums'],
  },
  metaRow: {
    marginTop: 24,
    flexDirection: 'row',
    gap: 8,
  },
  metaPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: 'rgba(255,255,255,0.08)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  metaText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '500',
  },
  missEmoji: {
    fontSize: 56,
  },
  missTitle: {
    color: '#fff',
    fontSize: 22,
    fontWeight: '600',
  },
  missBody: {
    color: 'rgba(255,255,255,0.65)',
    fontSize: 15,
    textAlign: 'center',
    paddingHorizontal: 16,
    lineHeight: 22,
  },
  cta: {
    alignSelf: 'center',
    backgroundColor: '#fff',
    paddingHorizontal: 36,
    paddingVertical: 14,
    borderRadius: 999,
    marginBottom: 32,
  },
  ctaText: {
    color: '#000',
    fontSize: 16,
    fontWeight: '600',
  },
});
