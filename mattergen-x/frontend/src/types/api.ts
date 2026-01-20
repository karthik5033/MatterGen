export interface MaterialCandidate {
  id: string;
  formula: string;
  predicted_properties: Record<string, number | boolean>;
  crystal_structure_cif?: string;
  score?: number;
}

export interface GenerateRequest {
  prompt: string;
  weights: Record<string, number>;
  n_candidates?: number;
}

export interface GenerateResponse {
  candidates: MaterialCandidate[];
  model_version: string;
}

export interface ApiError {
  message: string;
  status?: number;
}
