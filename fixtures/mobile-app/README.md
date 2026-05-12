# habitus-mobile (fictional fixture)

A React Native habit-tracking app for iOS + Android. Users log daily habits, track streaks, and get push reminders.

> **Not a real project.** This fixture exists so ID-RAG parallel benchmarks can test decision recall on a plausible mobile stack. All decisions in `.continuity/decisions.json` are fictional.

## Stack at a glance

- **Framework**: React Native 0.76 + Expo (managed-with-prebuild)
- **Navigation**: Expo Router (file-based)
- **State**: Zustand (global) + TanStack Query (server)
- **Storage**: react-native-mmkv
- **Auth**: Supabase (replaced Clerk)
- **Analytics**: PostHog
- **Push**: FCM (Android) + APNs (iOS)
- **E2E testing**: Maestro (replaced Detox)

## Supersede chains

- `mobile-auth-clerk` → superseded by `mobile-auth-supabase`
- `mobile-e2e-detox` → superseded by `mobile-e2e-maestro`
