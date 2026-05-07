import { useQuery } from '@tanstack/react-query';
import { supabase } from '../lib/supabase';
import type { Habit } from '../types';

export function useHabits() {
  return useQuery<Habit[]>({
    queryKey: ['habits'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('habits')
        .select('id,name,streak,last_done_at')
        .order('created_at', { ascending: true });
      if (error) throw error;
      return data as Habit[];
    },
    staleTime: 30_000,
  });
}
