import { FlatList, Text, View } from 'react-native';
import { useHabits } from '../hooks/useHabits';
import { useHabitStore } from '../stores/habitStore';
import { HabitRow } from '../components/HabitRow';

export function HabitListScreen() {
  const { data, isLoading } = useHabits();
  const selectedId = useHabitStore((s) => s.selectedId);

  if (isLoading) return <Text>Loading…</Text>;

  return (
    <View style={{ flex: 1 }}>
      <FlatList
        data={data}
        keyExtractor={(h) => h.id}
        renderItem={({ item }) => (
          <HabitRow habit={item} selected={item.id === selectedId} />
        )}
      />
    </View>
  );
}
