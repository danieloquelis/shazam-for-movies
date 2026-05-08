import React from 'react';
import { StyleSheet, View } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
  Easing,
  cancelAnimation,
} from 'react-native-reanimated';

type Props = {
  /** Size of the outer ring in points. */
  size?: number;
  /** Whether the ring is filling (recording active). */
  active: boolean;
  /** Total duration of one fill, in milliseconds. Reset when active flips. */
  durationMs: number;
};

/**
 * A thin progress ring that fills clockwise over `durationMs` while `active`,
 * and snaps back to empty when `active` flips back to false. Drawn entirely
 * with Views + transforms so we don't need react-native-svg.
 *
 * Implementation uses the classic "two semicircle masks" trick: each half of
 * the ring is rendered as a Pressable-shaped half-disc, and a coloured
 * rotating overlay sweeps from 0deg to 180deg over the right half, then
 * 180deg to 360deg over the left half.
 */
export function ScanRing({ size = 240, active, durationMs }: Props) {
  const progress = useSharedValue(0); // 0 → 1
  const half = size / 2;
  const stroke = 6;

  React.useEffect(() => {
    cancelAnimation(progress);
    if (active) {
      progress.value = 0;
      progress.value = withTiming(1, {
        duration: durationMs,
        easing: Easing.linear,
      });
    } else {
      progress.value = withTiming(0, { duration: 200 });
    }
  }, [active, durationMs, progress]);

  const rightStyle = useAnimatedStyle(() => {
    // First 50%: rotates 0 → 180 over the right half.
    const rotate = Math.min(progress.value, 0.5) * 360;
    return { transform: [{ rotate: `${rotate}deg` }] };
  });

  const leftStyle = useAnimatedStyle(() => {
    // Second 50%: opacity 0 until past midpoint, then rotates 180 → 360.
    const past = Math.max(progress.value - 0.5, 0);
    return {
      opacity: progress.value > 0.5 ? 1 : 0,
      transform: [{ rotate: `${past * 360 + 180}deg` }],
    };
  });

  return (
    <View
      pointerEvents="none"
      style={[
        styles.outer,
        {
          width: size,
          height: size,
          borderRadius: half,
          borderWidth: stroke,
        },
      ]}
    >
      {/* Right-half sweeping fill */}
      <View
        style={[
          styles.halfClip,
          {
            width: half + stroke,
            right: 0,
            borderTopRightRadius: half,
            borderBottomRightRadius: half,
          },
        ]}
      >
        <Animated.View
          style={[
            styles.halfFill,
            {
              width: stroke,
              height: size + stroke * 2,
              left: -stroke / 2,
              borderRadius: stroke,
              transformOrigin: 'right center',
            },
            rightStyle,
          ]}
        />
      </View>

      {/* Left-half sweeping fill (only visible past 50% progress) */}
      <View
        style={[
          styles.halfClip,
          {
            width: half + stroke,
            left: 0,
            borderTopLeftRadius: half,
            borderBottomLeftRadius: half,
          },
        ]}
      >
        <Animated.View
          style={[
            styles.halfFill,
            {
              width: stroke,
              height: size + stroke * 2,
              right: -stroke / 2,
              borderRadius: stroke,
              transformOrigin: 'left center',
            },
            leftStyle,
          ]}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  outer: {
    borderColor: 'rgba(255,255,255,0.25)',
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
  },
  halfClip: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    overflow: 'hidden',
  },
  halfFill: {
    position: 'absolute',
    top: -6,
    backgroundColor: '#ffffff',
  },
});
