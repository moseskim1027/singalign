export interface ComparisonExample {
  id: string;
  reference: string;
  baseline: string;
  aligned: string;
  disclosure: string;
  offset_seconds: number;
  selection_method: string;
}

export interface LatestComparison {
  run_id: string;
}

export interface ComparisonManifest {
  title: string;
  split: string;
  examples: ComparisonExample[];
  mlflow_experiment_id?: string;
  mlflow_run_id?: string;
}

export interface MetricDelta {
  confidence_level: number;
  lower: number;
  mean: number;
  upper: number;
}

export interface MetricSummary {
  baseline_mean: number;
  aligned_mean: number;
  delta: MetricDelta;
  wins: number;
  ties: number;
  losses: number;
}

export interface ComparisonSummary {
  split: string;
  examples: number;
  training_segment_seconds: number;
  comparison_segment_seconds: number;
  metric_direction: string;
  metrics: Record<string, MetricSummary>;
}

export interface MultiConditionMetric {
  log_mel_mse: number;
  log_mel_mae: number;
  spectral_convergence: number;
}

export interface MultiConditionRow {
  name: string;
  method: string;
  metrics: MultiConditionMetric | Record<string, { mean: number; lower: number; upper: number }>;
  effect_vs_anchor?: Record<string, { mean: number; lower: number; upper: number; standardized_effect?: number }>;
}

export interface MultiConditionReport {
  condition_count: number;
  example_count?: number;
  anchor?: string;
  conditions: MultiConditionRow[];
}
