import { create } from 'zustand';
import { MMKV } from 'react-native-mmkv';
import { persist, createJSONStorage } from 'zustand/middleware';

const storage = new MMKV({ id: 'habitus.store' });

type HabitStore = {
  selectedId: string | null;
  setSelected: (id: string | null) => void;
};

export const useHabitStore = create<HabitStore>()(
  persist(
    (set) => ({
      selectedId: null,
      setSelected: (id) => set({ selectedId: id }),
    }),
    {
      name: 'habits',
      storage: createJSONStorage(() => ({
        getItem: (k) => storage.getString(k) ?? null,
        setItem: (k, v) => storage.set(k, v),
        removeItem: (k) => storage.delete(k),
      })),
    },
  ),
);
