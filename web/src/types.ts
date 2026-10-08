export interface Metrics {
  roc_auc: number | null;
  pr_auc: number | null;
  precision_at_recall90: number | null;
  precision: number | null;
  recall: number | null;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  n: number;
  prevalence: number;
}

export interface Example {
  id: string;
  kind: "planet" | "eb" | "noise";
  label: 0 | 1;
  prob_planet: number;
  predicted_candidate: boolean;
  period_days: number;
  duration_hours: number;
  depth_ppm: number;
  snr: number;
  report: string;
  light_curve: { t: number[]; f: number[] };
  global_view: number[];
  local_view: number[];
}

export interface Demo {
  meta: {
    data_kind: string;
    n_examples_total: number;
    split_sizes: Record<string, number>;
    seed: number;
    device: string;
    epochs: number;
    best_epoch: number;
    torch: string;
    threshold_cnn: number;
    threshold_rf: number;
  };
  metrics: {
    cnn_default_threshold: Metrics;
    cnn_val_threshold: Metrics;
    rf_default_threshold: Metrics;
    rf_val_threshold: Metrics;
  };
  examples: Example[];
}
