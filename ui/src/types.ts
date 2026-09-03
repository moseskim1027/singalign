export interface ComparisonExample {
  id: string;
  reference: string;
  baseline: string;
  aligned: string;
  disclosure: string;
}

export interface ComparisonManifest {
  title: string;
  split: string;
  examples: ComparisonExample[];
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
  metric_direction: string;
  metrics: Record<string, MetricSummary>;
}
