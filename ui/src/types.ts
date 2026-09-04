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
