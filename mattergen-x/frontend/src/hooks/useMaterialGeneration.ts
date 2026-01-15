import { useState } from 'react';
import { ApiService } from '@/lib/api';
import { MaterialCandidate } from '@/types/api';

/**
 * Custom hook for managing material candidate generation.
 * Handles loading states, errors, and the resulting candidates list.
 */
export function useMaterialGeneration() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<MaterialCandidate[]>([]);

  const generate = async (prompt: string, weights: Record<string, number>) => {
    if (!prompt.trim()) return;

    setIsLoading(true);
    setError(null);
    
    try {
      const candidates = await ApiService.generateMaterials(prompt, weights, 3);
      setCandidates(candidates);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred during generation.");
    } finally {
      setIsLoading(false);
    }
  };

  return {
    generate,
    isLoading,
    error,
    candidates
  };
}
